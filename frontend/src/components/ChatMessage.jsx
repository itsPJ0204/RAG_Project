import React, { useContext } from "react";
import { motion } from "framer-motion";
import { ThemeContext } from "../themeContext";

export default function ChatMessage({ sender, text }) {
  const isUser = sender === "user";
  const { theme } = useContext(ThemeContext);

  const userBg =
    theme === "dark" ? "bg-blue-600 text-white" : "bg-accent text-white";
  const aiBg =
    theme === "dark"
      ? "bg-gray-800 text-white border border-gray-700 shadow-md"
      : "bg-white text-gray-900 shadow-sm border border-gray-200";

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex ${isUser ? "justify-end" : "justify-start"} my-2`}
    >
      <div
        className={`p-3 rounded-2xl max-w-xl ${
          isUser ? userBg + " rounded-br-none" : aiBg + " rounded-bl-none"
        }`}
      >
        {text}
      </div>
    </motion.div>
  );
}
