import React, { useState, useEffect, createContext, useContext, useRef } from "react";
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

// ---
// 1. API Configuration
// ---
// This will use your VITE_API_URL from the .env file when in production (e.g., on Hugging Face),
// but will default to your local Python server for development.

// This is the correct syntax for a Vite application.
const API_BASE_URL = import.meta.env.VITE_API_URL || "https://itspj0204-rag-for-movies.hf.space";

const API_ENDPOINT = `${API_BASE_URL}/api/chat`;

// ---
// 2. Context for Dark Mode
// ---
export const ThemeContext = createContext();

// ---
// 3. Re-usable Components
// ---
const Loader = () => {
  const { theme } = useContext(ThemeContext);
  return (
    <div className="flex justify-start my-3">
      <div className={`flex items-center justify-center p-3 rounded-2xl ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-100'}`}>
        <div className="flex gap-1">
          <div className={`w-2 h-2 rounded-full animate-bounce ${theme === 'dark' ? 'bg-gray-300' : 'bg-gray-600'}`}></div>
          <div className={`w-2 h-2 rounded-full animate-bounce delay-150 ${theme === 'dark' ? 'bg-gray-300' : 'bg-gray-600'}`}></div>
          <div className={`w-2 h-2 rounded-full animate-bounce delay-300 ${theme === 'dark' ? 'bg-gray-300' : 'bg-gray-600'}`}></div>
        </div>
      </div>
    </div>
  );
};

const ChatInput = ({ onSend, loading }) => {
  const [input, setInput] = useState("");
  const { theme } = useContext(ThemeContext);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;
    onSend(input);
    setInput("");
  };

  return (
    <div className={`p-4 ${theme === "dark" ? "bg-gray-800" : "bg-white"}`}>
      <form
        onSubmit={handleSubmit}
        className="flex items-center gap-3"
      >
        <input
          className={`flex-1 p-3 border rounded-xl outline-none focus:ring-2 focus:ring-blue-500 ${
            theme === "dark"
              ? "bg-gray-700 border-gray-600 text-white placeholder-gray-400"
              : "bg-white text-gray-900 border-gray-300"
          }`}
          placeholder="Ask about screenwriting, cameras, or a movie..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
        />
        <button
          type="submit"
          className="bg-blue-600 text-white p-3 rounded-xl hover:bg-blue-700 transition disabled:bg-gray-400"
          disabled={loading}
        >
          <Send size={18} />
        </button>
      </form>
    </div>
  );
};

const ChatMessage = ({ sender, text }) => {
  const isUser = sender === "user";
  const { theme } = useContext(ThemeContext);

  const userBg = "bg-blue-600 text-white";
  const aiBg =
    theme === "dark"
      ? "bg-gray-700 text-gray-100"
      : "bg-gray-100 text-gray-900";

  return (
    <div
      className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"} my-4`}
    >
      {!isUser && (
        <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-600'}`}>
          <Bot size={18} className={theme === 'dark' ? 'text-gray-300' : 'text-white'} />
        </div>
      )}
      <div
        className={`p-4 rounded-2xl max-w-lg md:max-w-xl lg:max-w-2xl shadow-sm ${
          isUser ? userBg + " rounded-br-none" : aiBg + " rounded-bl-none"
        }`}
      >
        {/* Use whitespace-pre-wrap to respect newlines and formatting */}
        <pre className="text-sm font-sans whitespace-pre-wrap">{text}</pre>
      </div>
      {isUser && (
        <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-600'}`}>
          <User size={18} className={theme === 'dark' ? 'text-gray-300' : 'text-white'} />
        </div>
      )}
    </div>
  );
};

// ---
// 4. Main App Component
// ---
export default function App() {
  const [chats, setChats] = useState(() => {
    try {
      const saved = localStorage.getItem("screenwriterAI_chats");
      return saved ? JSON.parse(saved) : [{ id: 1, title: "New Chat", messages: [] }];
    } catch (e) {
      console.error("Failed to parse chats from localStorage", e);
      return [{ id: 1, title: "New Chat", messages: [] }];
    }
  });
  
  const [activeChatId, setActiveChatId] = useState(chats[0].id);
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [editingChatId, setEditingChatId] = useState(null);
  const [newChatTitle, setNewChatTitle] = useState("");
  
  const [darkMode, setDarkMode] = useState(() => {
    const savedTheme = localStorage.getItem("screenwriterAI_theme");
    return savedTheme ? savedTheme === "dark" : false;
  });

  // FIX: This finds the active chat, or defaults to the first chat if not found
  const activeChat = chats.find((c) => c.id === activeChatId) || chats[0];
  const theme = darkMode ? "dark" : "light";
  const messagesEndRef = useRef(null);

  // --- Effects ---
  
  // Scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeChat?.messages]); // FIX: Was [messages], now updates when activeChat.messages changes

  // Save chats to localStorage
  useEffect(() => {
    localStorage.setItem("screenwriterAI_chats", JSON.stringify(chats));
  }, [chats]);

  // Save theme to localStorage and update <html> tag
  useEffect(() => {
    localStorage.setItem("screenwriterAI_theme", theme);
    document.documentElement.classList.toggle("dark", darkMode);
  }, [darkMode, theme]);
  
  // FIX: This effect was missing. It keeps the local `messages` state
  // in sync with the messages from the currently active chat.
  useEffect(() => {
    if (activeChat) {
      setMessages(activeChat.messages);
    } else {
      // Handle case where activeChat might be undefined (e.g., all chats deleted)
      setMessages([]);
    }
  // --- MY FIX --- 
  // The dependency array should only be `activeChatId` and `chats`.
  // `activeChat` is derived from them, so including it is redundant.
  }, [activeChatId, chats]);
  // --- END MY FIX ---

  // --- Chat Management ---
  
  // FIX: This state was missing. It holds the messages for the *current* view.
  const [messages, setMessages] = useState(activeChat ? activeChat.messages : []);

  const updateActiveChat = (newMessages) => {
    // FIX: This now updates BOTH the local `messages` state for the UI
    // and the permanent `chats` state for localStorage.
    setMessages(newMessages);
    setChats((prev) =>
      prev.map((chat) =>
        chat.id === activeChatId ? { ...chat, messages: newMessages } : chat
      )
    );
  };

  const handleSend = async (query) => {
    const userMessage = { sender: "user", text: query };
    // FIX: Uses the local `messages` state
    const newMessages = [...messages, userMessage];
    updateActiveChat(newMessages);
    setLoading(true);

    try {
      // FIX: Create history from newMessages, not activeChat.messages
      const chatHistory = newMessages.slice(0, -1).map(msg => ({
        sender: msg.sender,
        text: msg.text
      }));

      // FIX: Using the correct API_ENDPOINT, sending the correct JSON payload
      const response = await axios.post(API_ENDPOINT, { 
        query, 
        history: chatHistory 
      });
      
      // FIX: Reading from `response.data.response` not `response.data.answer`
      const aiMessage = { sender: "ai", text: response.data.response };
      updateActiveChat([...newMessages, aiMessage]);

    } catch (error) {
      console.error("Error fetching response:", error);
      const errorText = error.response
        ? `Error: ${error.response.status} - ${error.response.data.error || 'Unknown server error'}`
        : "Error fetching response. Check API URL and backend logs.";
      
      updateActiveChat([
        ...newMessages,
        { sender: "ai", text: `⚠️ ${errorText}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleNewChat = () => {
    const newId = Date.now();
    // --- MY FIX ---
    // Using 'New Chat' for consistency with the title used in `handleDeleteChat`
    const newChat = { id: newId, title: 'New Chat', messages: [] };
    // --- END MY FIX ---

    // If first chat, replace, otherwise add
    if (chats.length === 1 && chats[0].messages.length === 0) {
      setChats([newChat]);
    } else {
      setChats([...chats, newChat]);
    }
    setActiveChatId(newId);
    // FIX: Clear the local messages state
    setMessages([]);
  };

  const handleSelectChat = (id) => {
    setActiveChatId(id);
    // FIX: No need to setMessages here, the useEffect will handle it
  };

  const handleDeleteChat = (id) => {
    const updatedChats = chats.filter((chat) => chat.id !== id);
    
    if (updatedChats.length === 0) {
      // If no chats are left, create a new one
      const newId = Date.now();
      const newChat = { id: newId, title: `New Chat`, messages: [] };
      setChats([newChat]);
      setActiveChatId(newId);
      setMessages([]);
    } else {
      setChats(updatedChats);
      // If the active chat was deleted, set active to the first in the list
      if (activeChatId === id) {
        setActiveChatId(updatedChats[0].id);
        setMessages(updatedChats[0].messages);
      }
    }
  };


  const handleRenameChat = (id, title) => {
    if (!title.trim()) return; // Don't allow empty titles
    setChats((prev) =>
      prev.map((chat) => (chat.id === id ? { ...chat, title: title.trim() } : chat))
    );
    setEditingChatId(null);
  };

  return (
    <ThemeContext.Provider value={{ theme }}> 
      <div
        className={`flex h-screen transition-colors duration-300 ${
          darkMode ? "bg-gray-900 text-gray-100" : "bg-gray-50 text-gray-900"
        }`}
      >
        {/* --- Sidebar --- */}
        <div
          className={`${
            sidebarOpen ? "w-64" : "w-0"
          } border-r shadow-lg transition-all duration-300 flex flex-col overflow-hidden ${
            darkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"
          }`}
        >
          <div
            className={`flex items-center justify-between p-4 border-b ${
              darkMode ? "border-gray-700" : "border-gray-200"
            }`}
          >
            <h2 className={`font-semibold text-lg ${darkMode ? 'text-blue-400' : 'text-blue-600'}`}>
              Chat History
            </h2>
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
                className={`p-3 rounded-lg cursor-pointer mb-2 flex items-center justify-between group text-sm ${
                  chat.id === activeChatId
                    ? "bg-blue-600 text-white"
                    : darkMode
                    ? "text-gray-200 hover:bg-gray-700"
                    : "text-gray-800 hover:bg-gray-100"
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
                      if (e.key === "Escape") setEditingChatId(null);
                    }}
                    autoFocus
                    className={`w-full bg-transparent border-b outline-none ${
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
                    className="hover:text-red-500"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* --- Main Chat Area --- */}
        <div className="flex-1 flex flex-col h-screen">
          <header
            className={`flex items-center justify-between p-4 border-b shadow-sm ${
              darkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"
            }`}
          >
            <div className="flex items-center gap-3">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                // --- MY FIX ---
                // Removed `md:hidden` so the toggle is always visible
                className={`${darkMode ? 'text-gray-400 hover:text-white' : 'text-gray-500 hover:text-black'}`}
                // --- END MY FIX ---
              >
                <Menu size={22} />
              </button>
              <div className="flex items-center gap-2">
                <Rocket className={`${darkMode ? 'text-blue-400' : 'text-blue-600'}`} />
                <h1 className="text-xl font-semibold">ScriptCraft</h1>
              </div>
            </div>

            <button
              onClick={() => setDarkMode(!darkMode)}
              className={`p-2 rounded-lg ${
                darkMode ? "hover:bg-gray-700" : "hover:bg-gray-100"
              }`}
              title="Toggle theme"
            >
              {darkMode ? <Sun size={20} /> : <Moon size={20} />}
            </button>
          </header>

          {/* Chat messages */}
          <div className={`flex-1 overflow-y-auto p-4 ${darkMode ? 'bg-gray-900' : 'bg-gray-50'}`}>
            {/* FIX: Use the local `messages` state */}
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <Rocket size={48} className={`${darkMode ? 'text-blue-400' : 'text-blue-600'} mb-4`} />
                <h2 className="text-2xl font-semibold mb-2">Welcome to ScriptCraft</h2>
                <p className={`max-w-md ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  Ask about screenplay structure, camera techniques, or for creative inspiration!
                </p>
                <p className={`mt-4 text-sm ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                  Example: "Compare the opening scenes of Alien and Chinatown."
                </p>
              </div>
            )}
            
            {/* FIX: Use the local `messages` state */}
            {messages.map((msg, i) => (
              <ChatMessage
                key={i}
                sender={msg.sender}
                text={msg.text}
              />
            ))}
            {loading && <Loader />}
            <div ref={messagesEndRef} />
          </div>

          <ChatInput 
            onSend={handleSend} 
            loading={loading}
          />
        </div>
      </div>
    </ThemeContext.Provider>
  );
}