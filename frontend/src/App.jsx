import React, { useState, useEffect, createContext, useContext } from "react";
import axios from "axios";
import {
  Rocket,
  Plus,
  Menu,
  Trash2,
  Edit2,
  Moon,
  Sun,
  Send,
  User,
  Bot,
} from "lucide-react";
const API_URL = "https://rag-api.onrender.com/query";
export const ThemeContext = createContext();
const Loader = () => {
  return (
    <div className="flex justify-start my-3">
      <div className="flex gap-1">
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-150"></div>
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-300"></div>
      </div>
    </div>
  );
};

// ChatInput.jsx code
const ChatInput = ({ onSend }) => {
  const [input, setInput] = useState("");
  // This hook will now get the theme from the Provider in App
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
          className={`flex-1 p-3 border rounded-xl outline-none focus:ring-2 focus:ring-blue-500 ${
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
          className="bg-blue-600 text-white p-3 rounded-xl hover:bg-blue-700 transition"
        >
          <Send size={18} />
        </button>
      </form>
    </div>
  );
};

// ChatMessage.jsx code
const ChatMessage = ({ sender, text }) => {
  const isUser = sender === "user";
  // This hook will now get the theme from the Provider in App
  const { theme } = useContext(ThemeContext);

  // COLOR FIX: Made user chat box blue in both light and dark mode
  const userBg = "bg-blue-600 text-white";
    
  // COLOR FIX: Made AI chat box a contrasting color for visibility
  const aiBg =
    theme === "dark"
      ? "bg-gray-700 text-gray-100 border border-gray-600 shadow-md"
      : "bg-gray-100 text-gray-900 shadow-sm border border-gray-200";

  return (
    <div
      className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"} my-3`}
    >
      {!isUser && (
        <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-200'}`}>
          <Bot size={18} className={theme === 'dark' ? 'text-gray-300' : 'text-gray-600'} />
        </div>
      )}
      <div
        className={`p-3 rounded-2xl max-w-xl ${
          isUser ? userBg + " rounded-br-none" : aiBg + " rounded-bl-none"
        }`}
      >
        {/* Using <pre> to respect newlines from the AI */}
        <pre className="text-sm font-sans whitespace-pre-wrap">{text}</pre>
      </div>
       {isUser && (
        <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-200'}`}>
          <User size={18} className={theme === 'dark' ? 'text-gray-300' : 'text-gray-600'} />
        </div>
      )}
    </div>
  );
};


// ---
// Main App Component (Now provides the ThemeContext)
// ---
export default function App() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [chats, setChats] = useState(() => {
    const saved = localStorage.getItem("entrepreneurAI_chats");
    return saved ? JSON.parse(saved) : [{ id: 1, title: "New Chat", messages: [] }];
  });
  const [activeChatId, setActiveChatId] = useState(chats[0].id);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [editingChatId, setEditingChatId] = useState(null);
  const [newChatTitle, setNewChatTitle] = useState("");
  const [darkMode, setDarkMode] = useState(() => {
    const savedTheme = localStorage.getItem("entrepreneurAI_theme");
    return savedTheme ? savedTheme === "dark" : false;
  });

  const activeChat = chats.find((c) => c.id === activeChatId);
  
  // This is the value that will be passed to the context
  const theme = darkMode ? "dark" : "light";

  // 💾 Save chats and theme
  useEffect(() => {
    localStorage.setItem("entrepreneurAI_chats", JSON.stringify(chats));
  }, [chats]);

  useEffect(() => {
    localStorage.setItem("entrepreneurAI_theme", darkMode ? "dark" : "light");
    document.documentElement.classList.toggle("dark", darkMode);
  }, [darkMode]);

  const updateChat = (id, messages) => {
    setChats((prev) =>
      prev.map((chat) => (chat.id === id ? { ...chat, messages } : chat))
    );
  };

  const handleSend = async (query) => {
    const userMessage = { sender: "user", text: query };
    const newMessages = [...(activeChat?.messages || []), userMessage];
    updateChat(activeChatId, newMessages);
    setLoading(true);

    try {
      const response = await axios.post(API_URL, { query });
      
      const aiMessage = { sender: "ai", text: response.data.answer };
      updateChat(activeChatId, [...newMessages, aiMessage]);
    } catch (error) {
      console.error("Error fetching response:", error);
      updateChat(activeChatId, [
        ...newMessages,
        { sender: "ai", text: "⚠️ Error fetching response. (Check API_URL and backend logs)" },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleNewChat = () => {
    const newId = Date.now();
    const newChat = { id: newId, title: `Chat ${chats.length + 1}`, messages: [] };
    setChats([...chats, newChat]);
    setActiveChatId(newId);
    setMessages([]);
  };

  const handleSelectChat = (id) => setActiveChatId(id);

  const handleDeleteChat = (id) => {
    const updated = chats.filter((chat) => chat.id !== id);
    setChats(updated);
    if (activeChatId === id && updated.length > 0) {
      setActiveChatId(updated[0].id);
    } else if (updated.length === 0) {
      handleNewChat();
    }
  };

  const handleRenameChat = (id, title) => {
    setChats((prev) =>
      prev.map((chat) => (chat.id === id ? { ...chat, title } : chat))
    );
    setEditingChatId(null);
  };

  useEffect(() => {
    const chat = chats.find((c) => c.id === activeChatId);
    setMessages(chat ? chat.messages : []);
  }, [activeChatId, chats]);

  return (
    // Wrap entire app in the ThemeContext.Provider
    <ThemeContext.Provider value={{ theme }}> 
      <div
        className={`flex h-screen transition-colors duration-300 ${
          darkMode ? "bg-gray-900 text-gray-100" : "bg-gray-50 text-gray-900"
        }`}
      >
        {/* Sidebar */}
        <div
          className={`${
            sidebarOpen ? "w-64" : "w-0"
          } border-r shadow-md transition-all duration-300 flex flex-col overflow-hidden ${
            darkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"
          }`}
        >
          <div
            className={`flex items-center justify-between p-4 border-b ${
              darkMode ? "border-gray-700" : "border-gray-200"
            }`}
          >
            {/* COLOR FIX: Replaced 'text-accent' with high-contrast 'text-blue-600' */}
            <h2 className={`font-semibold text-lg ${darkMode ? 'text-blue-400' : 'text-blue-600'}`}>Chats</h2>
            <button
              onClick={handleNewChat}
              className={`hover:opacity-70 ${darkMode ? 'text-blue-400' : 'text-blue-600'}`}
              title="New Chat"
            >
              <Plus size={20} />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-2">
            {chats.map((chat) => (
              <div
                key={chat.id}
                className={`p-2 rounded-lg cursor-pointer mb-2 flex items-center justify-between group ${
                  // COLOR FIX: Forced high-contrast blue for active, visible text for inactive
                  chat.id === activeChatId
                    ? "bg-blue-600 text-white" // Active: Blue background, white text
                    : darkMode
                    ? "text-gray-200 hover:bg-gray-700" // Dark Inactive: Light text
                    : "text-gray-800 hover:bg-gray-100" // Light Inactive: Dark text
                }`}
              >
                {editingChatId === chat.id ? (
                  <input
                    type="text"
                    value={newChatTitle}
                    onChange={(e) => setNewChatTitle(e.target.value)}
                    onBlur={() => handleRenameChat(chat.id, newChatTitle)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") handleRenameChat(chat.id, newChatTitle);
                    }}
                    autoFocus
                    className={`w-full bg-transparent border-b outline-none text-sm ${
                      // COLOR FIX: High contrast border for rename
                      darkMode ? "border-gray-500 text-white" : "border-blue-600 text-black"
                    }`}
                  />
                ) : (
                  <div
                    className="flex-1 truncate"
                    onClick={() => handleSelectChat(chat.id)}
                  >
                    {chat.title}
                  </div>
                )}

                <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition">
                  <button
                    onClick={() => {
                      setEditingChatId(chat.id);
                      setNewChatTitle(chat.title);
                    }}
                    title="Rename"
                  >
                    <Edit2 size={14} />
                  </button>
                  <button
                    onClick={() => handleDeleteChat(chat.id)}
                    title="Delete"
                    className="text-red-500"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Main Chat Area */}
        <div
          className={`flex-1 flex flex-col max-w-3xl mx-auto border-l ${
            darkMode ? "border-gray-700" : "border-gray-200"
          }`}
        >
          <header
            className={`flex items-center justify-between p-4 border-b shadow-sm ${
              darkMode ? "bg-gray-800 border-gray-700" : "bg-white"
            }`}
          >
            <div className="flex items-center gap-2">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="text-gray-500 hover:text-gray-300 md:hidden"
              >
                <Menu size={22} />
              </button>
              {/* COLOR FIX: Replaced 'text-accent' with high-contrast 'text-blue-600' */}
              <Rocket className={`${darkMode ? 'text-blue-400' : 'text-blue-600'}`} />
              <h1 className="text-xl font-semibold">LogIQ</h1>
              <span
                className={`ml-2 text-sm ${
                  darkMode ? "text-gray-400" : "text-gray-500"
                }`}
              >
                — Your personal business assistant
              </span>
            </div>

            <button
              onClick={() => setDarkMode(!darkMode)}
              className={`p-2 rounded-lg ${
                darkMode ? "hover:bg-gray-700" : "hover:bg-gray-200"
              }`}
              title="Toggle theme"
            >
              {darkMode ? <Sun size={20} /> : <Moon size={20} />}
            </button>
          </header>

          {/* Chat messages */}
          {/* COLOR FIX: Removed the blue background from chat history, as requested by user in prev turn */}
          <div className={`flex-1 overflow-y-auto p-4 ${darkMode ? 'bg-gray-900' : 'bg-gray-50'}`}>
            {messages.map((msg, i) => (
              <ChatMessage
                key={i}
                sender={msg.sender}
                text={msg.text}
              />
            ))}
            {loading && <Loader />}
          </div>

          <ChatInput 
            onSend={handleSend} 
          />
        </div>
      </div>
    </ThemeContext.Provider>
  );
}

