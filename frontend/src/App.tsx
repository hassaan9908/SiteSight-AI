
import {
  useEffect,
  useState,
  type ChangeEvent,
  type FormEvent,
} from "react";
import "./index.css";

const API_BASE = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

type Detection = {
  class_id: number;
  label: string;
  confidence: number;
  bounding_box: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
};

type ImageStatistics = {
  original_width: number;
  original_height: number;
  channels: number;
  processed_width: number;
  processed_height: number;
  was_resized: boolean;
  mean_brightness: number;
};

type AnalysisResult = {
  success: boolean;
  filename: string | null;
  content_type: string;
  file_size_bytes: number;
  image: ImageStatistics;
  object_detection: {
    model: string;
    confidence_threshold: number;
    detection_count: number;
    detections: Detection[];
    annotated_image_data_url: string;
  };
  message: string;
};

function errorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "Something went wrong. Please try again.";
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [report, setReport] = useState("");
  const [error, setError] = useState("");
  const [reportError, setReportError] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return;
    }

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    return () => URL.revokeObjectURL(url);
  }, [file]);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const selectedFile = event.target.files?.[0] ?? null;

    setFile(selectedFile);
    setAnalysis(null);
    setReport("");
    setError("");
    setReportError("");
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!file) {
      setError("Please select an image first.");
      return;
    }

    const allowedTypes = ["image/jpeg", "image/png", "image/webp"];

    if (!allowedTypes.includes(file.type)) {
      setError("Please upload a JPEG, PNG, or WebP image.");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError("The maximum upload size is 10 MB.");
      return;
    }

    setError("");
    setReportError("");
    setAnalysis(null);
    setReport("");
    setIsAnalyzing(true);

    try {
      // Step 1: Upload the image and run YOLO inference.
      const formData = new FormData();
      formData.append("file", file);

      const analysisResponse = await fetch(
        `${API_BASE}/api/v1/analyze-image`,
        {
          method: "POST",
          body: formData,
        },
      );

      const analysisData = await analysisResponse.json();

      if (!analysisResponse.ok) {
        throw new Error(
          analysisData.detail || "Image analysis failed.",
        );
      }

      const result = analysisData as AnalysisResult;
      setAnalysis(result);
      setIsAnalyzing(false);

      // Step 2: Send the results to Gemini for a report.
      setIsGeneratingReport(true);

      try {
        const reportResponse = await fetch(
          `${API_BASE}/api/v1/generate-report`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              analysis: result,
            }),
          },
        );

        const reportData = await reportResponse.json();

        if (!reportResponse.ok) {
          throw new Error(
            reportData.detail || "Report generation failed.",
          );
        }

        setReport(reportData.report);
      } catch (reportFailure) {
        // Keep the YOLO results visible even if Gemini fails.
        setReportError(errorMessage(reportFailure));
      } finally {
        setIsGeneratingReport(false);
      }
    } catch (analysisFailure) {
      setError(errorMessage(analysisFailure));
    } finally {
      setIsAnalyzing(false);
    }
  }

  const detections = analysis?.object_detection.detections ?? [];

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="SiteSight AI home">
          <span className="brand-mark">S</span>
          <span>SiteSight <span className="brand-ai">AI</span></span>
        </a>

        <span className="topbar-label">
          Computer Vision Workspace
        </span>
      </header>

      <main className="main-content">
        <section className="hero">
          <div className="eyebrow">
            <span className="status-dot" />
            AI-POWERED IMAGE ANALYSIS
          </div>

          <h1>
            See more.
            <br />
            <span className="gradient-text">Understand better.</span>
          </h1>

          <p className="hero-description">
            Analyze an image with computer vision, inspect detected
            objects, and generate an AI-assisted report in seconds.
          </p>
        </section>

        <section className="workspace">
          <div className="panel upload-panel">
            <div className="section-heading">
              <div>
                <span className="section-kicker">01 / INPUT</span>
                <h2>Upload an image</h2>
              </div>
              <span className="step-number">01</span>
            </div>

            <form onSubmit={handleSubmit}>
              <label className="upload-zone">
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={handleFileChange}
                />

                {previewUrl ? (
                  <img
                    className="upload-preview"
                    src={previewUrl}
                    alt="Selected image preview"
                  />
                ) : (
                  <div className="upload-placeholder">
                    <span className="upload-icon">↑</span>
                    <strong>Choose an image to analyze</strong>
                    <span>JPEG, PNG, or WebP · Up to 10 MB</span>
                  </div>
                )}
              </label>

              {file && (
                <div className="file-meta">
                  <div className="file-badge">IMG</div>
                  <div className="file-details">
                    <strong>{file.name}</strong>
                    <span>{formatBytes(file.size)}</span>
                  </div>
                  <button
                    type="button"
                    className="text-button"
                    onClick={() => {
                      setFile(null);
                      setAnalysis(null);
                      setReport("");
                      setError("");
                      setReportError("");
                    }}
                  >
                    Remove
                  </button>
                </div>
              )}

              {error && (
                <div className="error-message" role="alert">
                  {error}
                </div>
              )}

              <button
                className="primary-button"
                type="submit"
                disabled={!file || isAnalyzing || isGeneratingReport}
              >
                {isAnalyzing
                  ? "Analyzing image..."
                  : isGeneratingReport
                    ? "Generating AI report..."
                    : "Analyze image"}
                <span aria-hidden="true">→</span>
              </button>
            </form>

            {/* <p className="privacy-note">
              Your API key stays on the backend, never in the browser.
            </p> */}
          </div>

          <div className="panel results-panel">
            <div className="section-heading">
              <div>
                <span className="section-kicker">02 / ANALYSIS</span>
                <h2>Detection results</h2>
              </div>
              <span className="step-number">02</span>
            </div>

            {!analysis ? (
              <div className="empty-state">
                <div className="empty-icon">⌕</div>
                <strong>Your results will appear here</strong>
                <p>
                  Upload an image and run the analysis to see detected
                  objects and their confidence scores.
                </p>
              </div>
            ) : (
              <div className="results-content">
                <div className="result-image-frame">
                  <img
                    src={
                      analysis.object_detection.annotated_image_data_url
                    }
                    alt="Image annotated with detected objects"
                  />
                </div>

                <div className="stats-grid">
                  <div className="stat-card">
                    <span>OBJECTS DETECTED</span>
                    <strong>
                      {analysis.object_detection.detection_count}
                    </strong>
                  </div>

                  <div className="stat-card">
                    <span>IMAGE DIMENSIONS</span>
                    <strong>
                      {analysis.image.original_width} ×{" "}
                      {analysis.image.original_height}
                    </strong>
                  </div>

                  <div className="stat-card">
                    <span>MODEL</span>
                    <strong>{analysis.object_detection.model}</strong>
                  </div>
                </div>

                <div className="subsection-heading">
                  <h3>Detected objects</h3>
                  <span>{detections.length} results</span>
                </div>

                {detections.length === 0 ? (
                  <p className="muted-message">
                    No objects were detected above the configured
                    confidence threshold. This does not mean the
                    image contains no objects.
                  </p>
                ) : (
                  <div className="detection-list">
                    {detections.map((detection, index) => (
                      <div
                        className="detection-row"
                        key={`${detection.class_id}-${index}`}
                      >
                        <div className="detection-symbol">◇</div>
                        <div className="detection-info">
                          <strong>{detection.label}</strong>
                          <span>
                            Box: {detection.bounding_box.x1},{" "}
                            {detection.bounding_box.y1} →{" "}
                            {detection.bounding_box.x2},{" "}
                            {detection.bounding_box.y2}
                          </span>
                        </div>
                        <span className="confidence">
                          {Math.round(detection.confidence * 100)}%
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                <div className="subsection-heading report-heading">
                  <h3>AI inspection report</h3>
                  <span className="ai-pill">GEMINI</span>
                </div>

                {isGeneratingReport && (
                  <div className="report-loading">
                    <span className="loading-spinner" />
                    Generating a report from the detection results...
                  </div>
                )}

                {reportError && (
                  <div className="warning-message" role="alert">
                    <strong>Report unavailable</strong>
                    <p>{reportError}</p>
                    <span>
                      Your object-detection results are still available.
                    </span>
                  </div>
                )}

                {report && (
                  <div className="report-content">
                    <pre>{report}</pre>
                    <p className="report-disclaimer">
                      AI-generated assistance only. Not a certified
                      safety inspection.
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </section>

        <footer className="footer">
          <span>SiteSight AI · Portfolio Project</span>
          <span>Computer vision + generative AI</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
