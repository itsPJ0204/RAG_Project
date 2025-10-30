import React, { useState, useContext } from "react";
import { Send } from "lucide-react";
import { ThemeContext } from "../ThemeContext";

export default function ChatInput({ onSend }) {
  const [input, setInput] = useState("");
  const { theme } = useContext(ThemeContext);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    onSend(input);
    setInput("");
  };

  return (
    <div className={theme === "dark" ? "bg-gray-900" : "bg-white"}>
      <form
        onSubmit={handleSubmit}
        className={`flex items-center gap-2 p-3 border-t sticky bottom-0 ${
          theme === "dark" ? "border-gray-700" : "border-gray-200"
        }`}
      >
        <input
          className={`flex-1 p-3 border rounded-xl outline-none focus:ring-2 focus:ring-accent ${
            theme === "dark"
              ? "bg-gray-800 border-gray-700 text-white placeholder-gray-400"
              : "bg-white text-gray-900 border-gray-200"
          }`}
          placeholder="Ask about startup funding, business models, scaling strategies..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <button
          type="submit"
          className="bg-accent text-white p-3 rounded-xl hover:opacity-90 transition"
        >
          <Send size={18} />
        </button>
      </form>
    </div>
  );
}
