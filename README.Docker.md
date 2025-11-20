# Docker Setup для sql-agent

## Быстрый старт

### Сборка и запуск с Docker

```bash
# Сборка образа
docker build -t sql-agent .

# Запуск контейнера
docker run -d \
  --name sql-agent \
  --network=host \
  sql-agent
```

### Запуск с Docker Compose (рекомендуется)

```bash
# Запуск всех сервисов (приложение + PostgreSQL)
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка сервисов
docker-compose down

# Остановка с удалением volumes
docker-compose down -v
```

## Структура файлов

- `Dockerfile` - образ приложения
- `docker-compose.yml` - оркестрация сервисов
- `.dockerignore` - исключения при сборке
- `requirements.txt` - Python зависимости

## Требования

Приложение требует:
1. **Ollama** - должен быть запущен и доступен (по умолчанию 127.0.0.1:11434)
2. **PostgreSQL** - для БД analytics_ai (должен быть запущен и доступен (по умолчанию 127.0.0.1:5432))
3. Внешние БД (MySQL, Vertica) - настраиваются через БД

## Переменные окружения

Убедитесь, что в файл `docker-compose.yml` передаются все необходимые параметры через gitlab-ci.yml:

```env
DB_CONFIG_PG_ANALYTICS_AI=postgresql://login:password@address:port/analytics_ai
# ... другие настройки
```

## API Endpoints

После запуска приложение доступно по адресу:
- API: http://localhost:9000
- Swagger UI: http://localhost:9000/docs
- Health check: http://localhost:9000/

## Troubleshooting

### Проверка логов

```bash
# Docker
docker logs sql-agent

# Docker Compose
docker-compose logs sql-agent
```

### Подключение к контейнеру

```bash
docker exec -it sql-agent /bin/bash
```
