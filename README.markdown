# Chainlit and LangGraph Data Layer Project

A PostgreSQL data layer for Chainlit applications integrated with LangGraph, featuring persistent storage for chat data and Google OAuth authentication. This project uses `uv` for dependency management and supports cloud storage uploads for file attachments.

Works with Chainlit >= `2.0.0` and LangGraph.

## Features

- PostgreSQL data layer defined in `prisma/schema.prisma`
- LangGraph integration for structured conversation workflows
- Cloud storage support for file uploads (AWS S3, Google Cloud Storage)
- Speech and text-to-text input/output (`app.py`)
- Google OAuth authentication for secure user access
- Persistent storage for threads, users, steps, elements, and feedback

## Prerequisites

- Python 3.8+
- `uv` for dependency management
- Docker and Docker Compose for local services
- Node.js for Prisma CLI
- Cloud provider credentials (AWS or Google Cloud) for production

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Install Dependencies

Using `uv`, install the dependencies defined in `pyproject.toml`:

```bash
uv sync
```

Additional dependencies for the database and cloud providers:

```bash
uv pip install asyncpg boto3 google-cloud-storage
```

### 3. Configure Environment Variables

Copy the example environment file and update it with your configuration:

```bash
cp .env.example .env
```

Edit `.env` to include:

- **Database Configuration**:
  ```env
  DATABASE_URL=postgresql://root:root@localhost:5432/postgres
  ```

- **Local S3 (Fake) Configuration** (for testing):
  ```env
  BUCKET_NAME=my-bucket
  APP_AWS_ACCESS_KEY=random-key
  APP_AWS_SECRET_KEY=random-key
  APP_AWS_REGION=eu-central-1
  DEV_AWS_ENDPOINT=http://localhost:4566
  ```

- **Google OAuth Authentication**:
  ```env
  OAUTH_GOOGLE_CLIENT_ID=your-google-client-id
  OAUTH_GOOGLE_CLIENT_SECRET=your-google-client-secret
  CHAINLIT_AUTH_SECRET=your-chainlit-auth-secret
  ```

### 4. Run Local Services

Start PostgreSQL and a fake S3 bucket using Docker Compose:

```bash
docker compose up -d
```

Apply the Prisma schema to the PostgreSQL database:

```bash
npx prisma migrate deploy
```

View your database schema and data:

```bash
npx prisma studio
```

### 5. Run the Application

- **Speech and Text-to-Text Mode**:
  ```bash
  uv run chainlit run app.py
  ```

The Chainlit app, powered by LangGraph, will connect to the PostgreSQL database and track threads, users, steps, elements, and feedback. File attachments are stored in the configured cloud storage (locally, the fake S3 bucket at `http://localhost:4566/my-bucket`).

## Authentication

This project includes Google OAuth authentication. Ensure the following are set in `.env`:

```env
OAUTH_GOOGLE_CLIENT_ID=your-google-client-id
OAUTH_GOOGLE_CLIENT_SECRET=your-google-client-secret
CHAINLIT_AUTH_SECRET=your-chainlit-auth-secret
```

Refer to Chainlit's [authentication documentation](https://docs.chainlit.io/authentication/overview) for setup details.

## Demo App

A demo app is available in the `demo_app/` directory. Follow the setup instructions there to explore a basic implementation with LangGraph and Chainlit.

## Production Deployment

For production, configure a production-grade PostgreSQL database with secure credentials and connect to a real cloud provider.

### AWS S3

```env
BUCKET_NAME=my-bucket
APP_AWS_ACCESS_KEY=your-access-key
APP_AWS_SECRET_KEY=your-secret-key
APP_AWS_REGION=eu-central-1
```

### Google Cloud Storage (GCS)

```env
BUCKET_NAME=my-test-bucket
APP_GCS_PROJECT_ID=your-project-id
APP_GCS_CLIENT_EMAIL=your-service-account-email
APP_GCS_PRIVATE_KEY=your-private-key
```

Create a service account with Storage Object Viewer and Creator/Admin permissions. Generate a key from the "Keys" tab in the Google Cloud Console.

## Live Deployment

This project has been deployed to Google Cloud. Check live [here](<your-deployment-url>).

## Troubleshooting

- Ensure all environment variables are correctly set in `.env`.
- Verify Docker services are running (`docker ps`).
- Check Prisma migrations have been applied (`npx prisma migrate deploy`).
- For cloud storage issues, confirm credentials and bucket names match your provider's configuration.
- Verify LangGraph workflows are correctly defined in your application code.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on the repository.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.