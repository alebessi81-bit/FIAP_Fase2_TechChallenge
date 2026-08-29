# Infraestrutura como codigo para o deploy no Cloud Run.
#
# Uso:
#   terraform init
#   terraform apply -var="projeto=SEU_PROJETO" -var="imagem=IMAGEM"
#
# IMPORTANTE: use um projeto DIFERENTE daquele onde esta a chave do
# Gemini. Habilitar billing no projeto da chave elimina a camada gratuita
# da API, e toda chamada passa a ser cobrada desde o primeiro token.

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

variable "projeto" {
  type        = string
  description = "ID do projeto GCP (nao o da chave do Gemini)"
}

variable "regiao" {
  type    = string
  default = "southamerica-east1" # Sao Paulo: menor latencia para o Brasil
}

variable "imagem" {
  type        = string
  description = "Imagem no Artifact Registry"
}

variable "gemini_api_key" {
  type      = string
  sensitive = true
  default   = ""
}

provider "google" {
  project = var.projeto
  region  = var.regiao
}

# A chave vai para o Secret Manager, nunca como variavel de ambiente em
# texto puro: variaveis aparecem no console e no historico de revisoes.
resource "google_secret_manager_secret" "gemini" {
  secret_id = "gemini-api-key"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "gemini" {
  count       = var.gemini_api_key == "" ? 0 : 1
  secret      = google_secret_manager_secret.gemini.id
  secret_data = var.gemini_api_key
}

resource "google_cloud_run_v2_service" "api" {
  name     = "triagem-diabetes"
  location = var.regiao

  template {
    # Escalabilidade automatica: e aqui que o requisito 2 do enunciado se
    # resolve. Sem trafego, zero instancias e custo zero.
    scaling {
      min_instance_count = 0
      max_instance_count = 10
    }

    # Uma instancia atende ate 40 requisicoes simultaneas. /predict roda
    # em milissegundos e nao bloqueia, entao subir instancia a cada
    # requisicao seria desperdicio.
    max_instance_request_concurrency = 40
    timeout                          = "60s"

    containers {
      image = var.imagem

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }

      env {
        name  = "LOG_LEVEL"
        value = "INFO"
      }

      env {
        name  = "GEMINI_MODELO"
        value = "gemini-3.5-flash-lite"
      }

      dynamic "env" {
        for_each = var.gemini_api_key == "" ? [] : [1]
        content {
          name = "GEMINI_API_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.gemini.secret_id
              version = "latest"
            }
          }
        }
      }

      # O Cloud Run so envia trafego depois que /health responde. Como o
      # modelo e treinado na subida, isso evita que a primeira requisicao
      # chegue antes de o modelo estar pronto.
      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 5
        period_seconds        = 3
        failure_threshold     = 10
      }

      liveness_probe {
        http_get {
          path = "/health"
        }
        period_seconds = 30
      }
    }
  }
}

output "url" {
  value = google_cloud_run_v2_service.api.uri
}
