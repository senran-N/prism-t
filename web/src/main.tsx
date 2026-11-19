import { Component, StrictMode } from "react";
import type { ErrorInfo, ReactNode } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import App from "./App";

class StudioErrorBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[Prism-T Studio]", error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <div
          style={{
            minHeight: "100vh",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: 24,
          }}
        >
          <div className="hero-card" style={{ maxWidth: 520, textAlign: "left" }}>
            <h2 style={{ marginBottom: 8 }}>工作台渲染异常</h2>
            <p className="mono" style={{ fontSize: 12, color: "#E11D48", wordBreak: "break-word" }}>
              {this.state.error.message}
            </p>
            <button
              type="button"
              className="btn primary"
              style={{ marginTop: 14 }}
              onClick={() => window.location.reload()}
            >
              重新加载工作台
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <StudioErrorBoundary>
      <App />
    </StudioErrorBoundary>
  </StrictMode>
);
