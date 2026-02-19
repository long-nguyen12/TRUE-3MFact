# Docker Setup Guide

## Prerequisites
- [Docker](https://www.docker.com/products/docker-desktop) installed and running
- [Docker Compose](https://docs.docker.com/compose/install/) (optional, for multi-container setups)

## Quick Start

### Build the Docker Image
```bash
docker build -t true-3mfact .
```

### Run the Container
```bash
docker run -it true-3mfact
```

## Advanced Usage

### With Volume Mounting
```bash
docker run -it -v $(pwd):/app true-3mfact
```

### Using Docker Compose
```bash
docker-compose up --build
```

## Configuration
- Modify `Dockerfile` for image customization
- Update `docker-compose.yml` for service configuration
- Set environment variables via `.env` file

## Troubleshooting
- Ensure Docker daemon is running: `docker info`
- Check image build logs: `docker build -t true-3mfact . --no-cache`
- View container logs: `docker logs <container-id>`

## Documentation
Refer to the project's `Dockerfile` and `docker-compose.yml` for detailed configuration.