# Weaviate Dashboard

A fast, modern dashboard for monitoring and managing Weaviate vector database instances.

![Weaviate Dashboard](https://raw.githubusercontent.com/weaviate/weaviate-logo/main/Weaviate-Horizontal-Logo-Blue-RGB.svg)

## Features

- 📊 Overview of your Weaviate instance with key metrics
- 🔎 Explore classes and their properties
- 📝 View and manage objects within classes
- 🧹 Delete unwanted classes
- 📈 Run inspections to analyze database performance and storage
- 🛠️ Fast and responsive UI with REST API backend

## Requirements

- Python 3.7+
- Weaviate instance (local or remote)
- Required Python packages (see Installation)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd weaviate_dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Weaviate API key (if needed):
```bash
# Create a .env file
echo "WEAVIATE_API_KEY=your-api-key" > .env
echo "WEAVIATE_URL=http://your-weaviate-instance:8080" >> .env
```

## Usage

Start the dashboard with:

```bash
python dashboard_of_weaviate/new_dashboard.py
```

Then open your browser and navigate to: `http://localhost:8000`

## API Documentation

The REST API documentation is available at: `http://localhost:8000/docs`

## Configuration

You can configure the following environment variables:

- `WEAVIATE_URL`: The URL of your Weaviate instance (default: http://localhost:8090)
- `WEAVIATE_API_KEY`: Your Weaviate API key (if authentication is enabled)

## Development

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT 