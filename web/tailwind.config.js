/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Chrome acromático vía CSS vars (voltean light/dark). `brand` mapea a
        // ink-high a propósito: el acento es tinta de alto contraste, NO un color.
        void: "rgb(var(--canvas) / <alpha-value>)",
        panel: "rgb(var(--panel) / <alpha-value>)",
        raised: "rgb(var(--raised) / <alpha-value>)",
        hairline: "rgb(var(--hairline) / <alpha-value>)",
        ink: {
          high: "rgb(var(--ink-high) / <alpha-value>)",
          mid: "rgb(var(--ink-mid) / <alpha-value>)",
          low: "rgb(var(--ink-low) / <alpha-value>)",
        },
        brand: {
          DEFAULT: "rgb(var(--ink-high) / <alpha-value>)",
          deep: "rgb(var(--ink-mid) / <alpha-value>)",
        },
        // Único color saturado de la app: el estado del conductor (fijo, no themea).
        state: {
          alert: "#2DD4BF",
          drowsy: "#FFB020",
          distracted: "#FF6B81",
          nosignal: "#94A3B8",
        },
      },
      fontFamily: {
        // Space Grotesk is the only web font loaded (index.html). `sans` is the
        // global default; `body` aliases the same loaded stack so UI text never
        // silently falls back to an unloaded family.
        sans: ['"Space Grotesk"', "system-ui", "sans-serif"],
        body: ['"Space Grotesk"', "system-ui", "sans-serif"],
        mono: ['ui-monospace', "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
      borderRadius: {
        "2xl": "1rem",
      },
      keyframes: {
        breathe: {
          "0%, 100%": { transform: "scale(1)", opacity: "0.55" },
          "50%": { transform: "scale(1.06)", opacity: "1" },
        },
        "fade-in": {
          from: { opacity: "0", transform: "translateY(6px)" },
          to: { opacity: "1", transform: "none" },
        },
      },
      animation: {
        breathe: "breathe 4s cubic-bezier(0.22,1,0.36,1) infinite",
        "fade-in": "fade-in 0.4s cubic-bezier(0.22,1,0.36,1) both",
      },
    },
  },
  plugins: [],
};
