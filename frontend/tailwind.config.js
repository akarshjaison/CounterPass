/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        pitch: {
          light: '#3f6212',  // light grass
          dark: '#1e3a1e',   // dark pitch
          base: '#224222',   // soccer board green
        },
        sports: {
          neon: '#0df27b',    // Neon green for observed options
          inferred: '#38bdf8',// Cyan for temporally inferred options
          intercept: '#ff4d4d',// Red for intercepted/risk lanes
          warning: '#fbbf24', // Yellow for unsafe lanes
        },
        slate: {
          950: '#070a13',
          900: '#0d1326',
          800: '#1a233d',
          700: '#2c3a5e',
        }
      },
      fontFamily: {
        sans: ['Outfit', 'Inter', 'sans-serif'],
      },
      boxShadow: {
        glass: '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
        'glass-glow': '0 8px 32px 0 rgba(13, 242, 123, 0.15)',
      },
      backdropBlur: {
        glass: '12px',
      }
    },
  },
  plugins: [],
}
