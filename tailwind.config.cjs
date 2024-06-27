const defaultTheme = require("tailwindcss/defaultTheme");
const colors = require("tailwindcss/colors");

module.exports = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
    "./layouts/**/*.{js,ts,jsx,tsx}",
    "./content/**/*.{md,mdx}",
    "./node_modules/@portaljs/core/dist/*.js",
    "./node_modules/@portaljs/core/*.js",
  ],
  darkMode: "class",
  theme: {
    extend: {
      // support wider width for large screens >1440px eg. in hero
      maxWidth: {
        "8xl": "88rem",
      },
      fontFamily: {
         sans: ["Montserrat","ui-sans-serif", ...defaultTheme.fontFamily.sans],
        headings: ["Kanit","-apple-system", ...defaultTheme.fontFamily.sans],
      },
      colors: {
        background: {
          DEFAULT: "#EAE7B1",
          dark: "#8F4309",
        },
        primary: {
          DEFAULT: "#3C6255",
          dark: "#A6BB8D",
        },
        secondary: {
          DEFAULT: "#22c55e",
          dark: "#22c55e",
        },
      },
    },
  },
  /* eslint global-require: off */
  plugins: [
    require("@tailwindcss/typography")
  ],
};
