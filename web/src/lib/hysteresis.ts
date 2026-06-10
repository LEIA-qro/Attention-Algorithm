import type { DriverState } from "./types";

export const SEV: Record<DriverState, number> = {
  Alert: 0,
  Drowsy: 1,
  Distracted: 2,
};

export interface HysteresisOptions {
  window: number;
  escalate: number;
  deescalate: number;
  emaAlpha: number;
}

const DEFAULTS: HysteresisOptions = {
  window: 5,
  escalate: 3,
  deescalate: 4,
  emaAlpha: 0.4,
};

/**
 * Asymmetric debounce over a sliding window of raw classifications.
 * Escalates fast to a more severe state (safety), relaxes slowly back to Alert
 * (avoids flicker). Confidence is smoothed with an EMA.
 */
export class HysteresisFilter {
  private buffer: DriverState[] = [];
  private current: DriverState = "Alert";
  private ema = 0;
  private readonly opts: HysteresisOptions;

  constructor(opts: Partial<HysteresisOptions> = {}) {
    this.opts = { ...DEFAULTS, ...opts };
  }

  push(sample: { state: DriverState; confidence: number }): {
    state: DriverState;
    confidence: number;
  } {
    this.buffer.push(sample.state);
    if (this.buffer.length > this.opts.window) this.buffer.shift();
    this.ema =
      this.ema === 0
        ? sample.confidence
        : this.opts.emaAlpha * sample.confidence + (1 - this.opts.emaAlpha) * this.ema;

    const counts = this.buffer.reduce<Record<string, number>>((acc, s) => {
      acc[s] = (acc[s] ?? 0) + 1;
      return acc;
    }, {});

    const candidates = (Object.keys(SEV) as DriverState[])
      .filter((s) => s !== this.current)
      .sort((a, b) => SEV[b] - SEV[a]);

    for (const s of candidates) {
      const needed = SEV[s] > SEV[this.current] ? this.opts.escalate : this.opts.deescalate;
      if ((counts[s] ?? 0) >= needed) {
        this.current = s;
        break;
      }
    }

    return { state: this.current, confidence: this.ema };
  }

  /** Fuerza el estado actual (p.ej. cuando el server ya confirmó un estado sostenido,
   * que es una garantía más fuerte que el debounce del cliente). */
  force(state: DriverState) {
    this.current = state;
    this.buffer = Array(this.opts.window).fill(state);
  }
}
