/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: "#1F2937", // dark slate
        accent: "#2563EB", // business blue
        light: "#F9FAFB",
      },
    },
  },
  plugins: [],
  darkMode: "class", 
};
