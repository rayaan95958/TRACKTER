import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        'yellow-400': '#FFBF00',
        'yellow-600': '#d0a00e',
        'gray-400': '#b3bbc2',
        'gray-600': '#475569',
        'gray-300': '#cbd5e0',
      },
      boxShadow: {
        'md': '0 0 0 2px #cbd5e0', // Light gray border shadow
      },
      height: {
        '15': '3.75rem', // Height of 15 units
      },
      resize: {
        'x-none': 'vertical', // Custom utility for disabling x-axis resize
      },
    },
  },
  plugins: [],
};

export default config;
