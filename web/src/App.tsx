import { Component, lazy, Suspense, type ReactNode } from "react";
import { createBrowserRouter, Navigate, RouterProvider } from "react-router-dom";
import { getLang, pick, useLang } from "@/lib/prefs";

const Landing = lazy(() => import("./pages/Landing"));
const Conductor = lazy(() => import("./pages/Conductor"));
const Manager = lazy(() => import("./pages/Manager"));
const TripDetail = lazy(() => import("./pages/TripDetail"));
const Sensors = lazy(() => import("./pages/Sensors"));

function BrandMark({ size = 48 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M3 14c2.8-4.2 6-6.3 9-6.3s6.2 2.1 9 6.3" />
      <circle cx="12" cy="14" r="2.1" fill="currentColor" stroke="none" />
    </svg>
  );
}

function Splash() {
  const t = pick(useLang());
  return (
    <div className="grid h-full place-items-center bg-void">
      <div className="flex animate-breathe flex-col items-center gap-3 text-brand">
        <BrandMark />
        <span className="text-xs uppercase tracking-[0.3em] text-ink-low">
          {t("Cargando", "Loading")}
        </span>
      </div>
    </div>
  );
}

class ErrorBoundary extends Component<{ children: ReactNode }, { hasError: boolean }> {
  state = { hasError: false };
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  render() {
    if (this.state.hasError) {
      const t = pick(getLang());
      return (
        <div className="grid h-full place-items-center bg-void p-6 text-center">
          <div className="flex max-w-sm flex-col items-center gap-5">
            <div className="text-ink-low">
              <BrandMark size={40} />
            </div>
            <div>
              <p className="font-sans text-xl font-bold text-ink-high">
                {t("Algo falló", "Something went wrong")}
              </p>
              <p className="mt-2 text-sm text-ink-mid">
                {t(
                  "Ocurrió un error inesperado. Recarga para volver a intentar.",
                  "An unexpected error occurred. Reload to try again.",
                )}
              </p>
            </div>
            <button
              onClick={() => window.location.reload()}
              className="rounded-full bg-brand px-6 py-2.5 font-semibold text-void transition-transform duration-150 hover:scale-[1.03] active:scale-100"
            >
              {t("Recargar", "Reload")}
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

const router = createBrowserRouter([
  { path: "/", element: <Landing /> },
  { path: "/conductor", element: <Conductor /> },
  { path: "/manager", element: <Manager /> },
  { path: "/manager/viajes/:id", element: <TripDetail /> },
  { path: "/sensores", element: <Sensors /> },
  { path: "*", element: <Navigate to="/" replace /> },
]);

export default function App() {
  return (
    <ErrorBoundary>
      <Suspense fallback={<Splash />}>
        <RouterProvider router={router} />
      </Suspense>
    </ErrorBoundary>
  );
}
