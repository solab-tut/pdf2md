# pdf2md

PDF to Markdown converter powered by Vision LLM. Converts PDF pages to structured Markdown with figure/table extraction, math formula support (LaTeX), and bilingual (Japanese/English) processing.

## Requirements

- Docker & Docker Compose
- NVIDIA GPU with CUDA support (for Ollama)
- NVIDIA Container Toolkit (`nvidia-docker`)

## Quick Start

```bash
# 1. Clone and enter the directory
git clone git@github.com:solab-tut/pdf2md.git
cd pdf2md

# 2. Create .env from template
cp .env.example .env

# 3. Start services
docker compose up -d

# 4. Pull the vision model (first time only, ~21GB)
docker compose exec ollama ollama pull qwen2.5vl:32b

# 5. Open in browser
open http://localhost:3200
```

## Architecture

```
┌─────────────────────────────────────────┐
│  Browser  http://localhost:3200         │
│  ┌───────────────────────────────────┐  │
│  │  Web UI (single-page app)         │  │
│  └──────────────┬────────────────────┘  │
└─────────────────┼───────────────────────┘
                  │ API
┌─────────────────┼───────────────────────┐
│  pdf2md         │            port 3200  │
│  ┌──────────────┴────────────────────┐  │
│  │  Flask backend                    │  │
│  │  - PDF rasterization (PyMuPDF)    │  │
│  │  - LLM provider abstraction       │  │
│  │  - Asset extraction pipeline      │  │
│  └──────────────┬────────────────────┘  │
│                 │                        │
│  ┌──────────────┴────────────────────┐  │
│  │  Providers                        │  │
│  │  ├ ollama (default)               │  │
│  │  └ azure_openai (optional)        │  │
│  └──────────┬───────────┬────────────┘  │
└─────────────┼───────────┼───────────────┘
              │           │
   ┌──────────┴──┐   ┌────┴──────────────┐
   │ pdf2md-     │   │ Azure OpenAI      │
   │ ollama      │   │ (optional)        │
   │ port 3201   │   └───────────────────┘
   └─────────────┘
```

### Services

| Container | Port | Description |
|---|---|---|
| `pdf2md` | 3200 | Web UI + API server (Flask) |
| `pdf2md-ollama` | 3201 | Local LLM server (Ollama, GPU) |

## Usage

### Basic Conversion

1. Open `http://localhost:3200`
2. Select a PDF file
3. Configure options (see below)
4. Click **Convert**
5. View results in Markdown or Preview mode
6. Click **Download** to save as ZIP (Markdown + assets)

### Conversion Options

| Option | Values | Description |
|---|---|---|
| **Lang** | Japanese / English | OCR language and prompt rules |
| **Model** | dropdown | Vision LLM model. Shows provider in parentheses |
| **Pages** | e.g. `1-3,5,8` | Page range to convert. Empty = all pages |
| **DPI** | 150 / 200 / 300 | Page rasterization resolution. Higher = better quality, slower |
| **Thinking** | on / off | Enable model reasoning (slower but may improve accuracy) |
| **Figures** | MD only / Extract as image | `MD only`: describe figures in text. `Extract as image`: crop and embed as PNG |
| **Tables** | MD only / Complex as image / All as image | `MD only`: all tables as Markdown pipes. `Complex as image`: simple tables as MD, complex (merged cells) as image. `All as image`: all tables as images |
| **Also save as image** | on / off | When tables are rendered as Markdown, also save a cropped image copy |

### Output

- **Markdown view**: Raw Markdown source, editable
- **Preview view**: Rendered HTML with KaTeX math formulas and inline images
- **Download**: ZIP archive containing:
  - `document.md` — Full Markdown
  - `assets/` — Extracted figures and tables (PNG)
  - `assets/p{N}-page.jpg` — Full page images

### Reload & Re-edit

Previously downloaded ZIP files can be reloaded via the **Reload** field to continue editing, re-crop figures, or adjust content.

### Manual Crop

Click any page thumbnail in the **Pages** strip to open the manual crop tool. Select a region, choose the asset type (figure/table), and add a caption. The cropped image is inserted into the Markdown at the cursor position.

## Configuration

### Environment Variables (.env)

Copy `.env.example` to `.env` and adjust as needed:

```bash
# Local LLM
OLLAMA_URL=http://ollama:11434      # Ollama endpoint (container name)
PDF2MD_MODEL=qwen2.5vl:32b          # Default model

# Image processing
PDF2MD_DEFAULT_DPI=150               # Default page rasterization DPI
PDF2MD_MAX_IMAGE_EDGE=1800           # Max pixel dimension before downscaling
PDF2MD_JPEG_QUALITY=75               # JPEG quality for OCR images (0-100)
PDF2MD_PAGE_ASSET_DPI=200            # Full-page export DPI
PDF2MD_PAGE_ASSET_JPEG_QUALITY=90    # Full-page export JPEG quality

# LLM parameters
PDF2MD_MAX_TOKENS=8192               # Max response tokens per page
PDF2MD_OLLAMA_KEEP_ALIVE=15m         # Model keep-alive duration
PDF2MD_OLLAMA_TIMEOUT=600            # HTTP timeout in seconds
PDF2MD_DEFAULT_THINKING=off          # Default thinking mode (on/off)
```

### GPU Configuration

By default, GPU device `1` is used. To change this, edit `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0']    # Change to your GPU ID
          capabilities: [gpu]
```

To check available GPUs: `nvidia-smi`

### Azure OpenAI (Optional)

To enable Azure OpenAI models alongside local Ollama, uncomment and fill in the Azure section in `.env`:

```bash
AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_DEPLOYMENTS=gpt-4o,gpt-4o-mini    # Comma-separated deployment names
AZURE_OPENAI_API_VERSION=2024-10-21
```

Azure models will appear in the model dropdown with `(azure)` suffix. You can switch between local and Azure models per conversion.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/pdf2md/health` | Health check |
| `GET` | `/api/pdf2md/models` | List available vision models (all providers) |
| `POST` | `/api/pdf2md/convert` | Convert PDF to Markdown (SSE stream) |

### POST /api/pdf2md/convert

**Request**: `multipart/form-data`

| Field | Type | Default | Description |
|---|---|---|---|
| `pdf` | file | (required) | PDF file |
| `lang` | string | `ja` | `ja` or `en` |
| `model` | string | from .env | Model name |
| `provider` | string | `ollama` | Provider name (`ollama` or `azure`) |
| `dpi` | int | `150` | Rasterization DPI |
| `thinking` | string | `off` | `on` or `off` |
| `extract_figures` | string | `off` | `on` or `off` |
| `tables` | string | `md` | `md`, `complex`, or `all` |
| `table_also_image` | string | `off` | `on` or `off` |
| `pages` | string | (all) | Page range, e.g. `1-3,5,8` |

**Response**: Server-Sent Events (SSE) stream

```
data: {"type": "start", "total_pages": 11, "format": "md"}
data: {"type": "page", "page": 1, "page_label": 1, "content": "# Title\n...", "metrics": {...}, "assets": [...]}
data: {"type": "asset", "filename": "p1-figure1.png", "data": "<base64>"}
data: {"type": "done"}
```

## Development

Source files are volume-mounted into the container for live editing:

```
backend/app.py        → /app/app.py
backend/static/       → /app/static/
backend/providers/    → /app/providers/
```

Changes to Python files require a container restart:

```bash
docker compose restart pdf2md
```

Changes to `index.html` take effect on browser reload.

## License

MIT
