import { useEffect } from "react";

type SentinelLike = { release: () => Promise<void> };

export function useWakeLock(active: boolean) {
  useEffect(() => {
    if (!active || !("wakeLock" in navigator)) return;
    let sentinel: SentinelLike | null = null;
    let released = false;

    const acquire = async () => {
      if (released || sentinel) return;
      try {
        const next = await (
          navigator as unknown as { wakeLock: { request: (t: string) => Promise<SentinelLike> } }
        ).wakeLock.request("screen");
        // El effect pudo limpiarse mientras esperábamos: no dejar el lock colgado.
        if (released) {
          void next.release();
          return;
        }
        sentinel = next;
      } catch {
        /* user denied or not allowed; ignore */
      }
    };

    const onVisibility = () => {
      // Al ir a background el lock se libera solo; al volver hay que re-pedirlo.
      if (document.visibilityState === "hidden") {
        sentinel = null;
      } else if (!released) {
        void acquire();
      }
    };

    void acquire();
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      released = true;
      document.removeEventListener("visibilitychange", onVisibility);
      void sentinel?.release();
    };
  }, [active]);
}
