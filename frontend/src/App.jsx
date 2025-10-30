import React, { useState, useEffect } from "react";
import axios from "axios";
import ChatMessage from "./components/ChatMessage";
import ChatInput from "./components/ChatInput";
import Loader from "./components/Loader";
import { Rocket, Plus, Menu, Trash2, Edit2, Moon, Sun } from "lucide-react";

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
      const response = await axios.post("http://127.0.0.1:5000/query", { query });
      const aiMessage = { sender: "ai", text: response.data.answer };
      updateChat(activeChatId, [...newMessages, aiMessage]);
    } catch (error) {
      updateChat(activeChatId, [
        ...newMessages,
        { sender: "ai", text: "⚠️ Error fetching response." },
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
          <h2 className="font-semibold text-lg text-accent">Chats</h2>
          <button
            onClick={handleNewChat}
            className="text-accent hover:text-accent/70"
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
                chat.id === activeChatId
                  ? "bg-accent text-white"
                  : darkMode
                  ? "hover:bg-gray-700 text-gray-200"
                  : "hover:bg-gray-100 text-gray-800"
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
                    darkMode ? "border-gray-500 text-white" : "border-accent"
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
            <Rocket className="text-accent" />
            <h1 className="text-xl font-semibold">EntrepreneurAI</h1>
            <span
              className={`ml-2 text-sm ${
                darkMode ? "text-gray-400" : "text-gray-500"
              }`}
            >
              — Your startup strategy assistant
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
        <div className="flex-1 overflow-y-auto p-4">
          {messages.map((msg, i) => (
            <ChatMessage
              key={i}
              sender={msg.sender}
              text={msg.text}
              darkMode={darkMode}
            />
          ))}
          {loading && <Loader />}
        </div>

        <ChatInput onSend={handleSend} darkMode={darkMode} />
      </div>
    </div>
  );
}
