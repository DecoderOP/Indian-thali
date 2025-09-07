"use client";

import { useState, useEffect, useCallback } from "react";
import MessageBubble from "./MessageBubble";
import InputBox from "./InputBox";
import Sidebar from "./Sidebar";
import { generatePlan, UserProfileData } from "../lib/api";
import { useConversations } from "../lib/useConversations";
import { DietPlanResponse } from "../lib/types";

import { Message } from "../lib/chatTypes";

export default function ChatWindow() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const { conversations, saveConversation, deleteConversation } =
    useConversations();
  const [activeConversationId, setActiveConversationId] = useState<
    string | null
  >(null);

  // Save conversation when messages change
  useEffect(() => {
    if (messages.length >= 2) {
      // Only save if there's at least one exchange
      saveConversation(messages, activeConversationId);
    }
    // We're intentionally excluding saveConversation from dependencies
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages, activeConversationId]); // Re-run when messages or activeId change

  // Function to clear the conversation
  function startNewConsultation() {
    setMessages([]);
    setActiveConversationId(null);
  }

  // Function to load a conversation
  function loadConversation(conversationId: string) {
    if (conversations[conversationId]) {
      // Don't reload if it's already the active conversation
      if (activeConversationId === conversationId) return;

      setMessages([...conversations[conversationId].messages]);
      setActiveConversationId(conversationId);
    }
  }

  // Function to handle conversation deletion
  function handleDeleteConversation(id: string) {
    deleteConversation(id);
    if (activeConversationId === id) {
      setMessages([]);
      setActiveConversationId(null);
    }
  }

  async function handleSend(userInput: string) {
    setMessages((prev) => [...prev, { sender: "user", text: userInput }]);
    setLoading(true);

    try {
      // Create a user profile from the input
      // For simplicity, we're parsing the first line as the disease
      // and any comma-separated parts as symptoms
      const lines = userInput.split("\n").filter((line) => line.trim());
      const disease = lines[0] || userInput;

      // Extract symptoms from the rest of the text or use the main text
      // In a real app, you'd have proper form inputs for these values
      const symptomsText =
        lines.length > 1 ? lines.slice(1).join(" ") : userInput;
      const symptoms = symptomsText.split(",").map((s) => s.trim());

      // Default values for age and gender - in a real app,
      // you would collect these from the user
      const userProfile = {
        disease: disease,
        symptoms: symptoms,
        age: 35, // Default age
        gender: "Not specified", // Default gender
      };

      const response = (await generatePlan(userProfile)) as DietPlanResponse;

      // Display the plan from the backend
      setMessages((prev) => [
        ...prev,
        {
          sender: "ai",
          text:
            response.plan ||
            "I couldn't generate a personalized plan. Please try again.",
        },
      ]);
    } catch (error) {
      console.error("Error generating plan:", error);
      setMessages((prev) => [
        ...prev,
        {
          sender: "ai",
          text: "Something went wrong generating your personalized diet plan. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  // State for sidebar measurements and screen size
  const [sidebarWidth, setSidebarWidth] = useState<number>(256); // Default width
  const [isMobile, setIsMobile] = useState(false);

  // Check if we're on client-side and set initial mobile state
  useEffect(() => {
    setIsMobile(window.innerWidth < 768);

    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Callback to update main content padding when sidebar width changes
  // Using useCallback to memoize the function and prevent unnecessary re-renders
  const handleSidebarResize = useCallback(
    (width: number) => {
      // Only update if the width has actually changed
      if (width !== sidebarWidth) {
        setSidebarWidth(width);
      }
    },
    [sidebarWidth]
  );

  return (
    <div className="flex flex-col h-screen bg-[var(--background)]">
      {/* Sidebar Component */}
      <Sidebar
        onNewConsultation={startNewConsultation}
        onLoadConversation={loadConversation}
        onDeleteConversation={handleDeleteConversation}
        conversations={conversations}
        activeConversationId={activeConversationId}
        onResize={handleSidebarResize}
      />

      {/* Main content - chat interface */}
      <div
        className="flex flex-col flex-1 md:transition-all md:duration-300"
        style={{ paddingLeft: !isMobile ? `${sidebarWidth}px` : "0px" }}
      >
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center max-w-3xl mx-auto px-6">
                <div className="mb-3">
                  <span className="inline-block text-[var(--secondary)] font-bold text-2xl">
                    Indian
                  </span>
                  <span className="inline-block text-[var(--primary)] italic font-bold text-2xl">
                    {" "}
                    Thali
                  </span>
                </div>
                <p className="text-[var(--neutral-content)] mb-8 max-w-lg mx-auto">
                  Discover personalized Indian Thali recommendations based on
                  your wellness needs
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-8 mx-auto">
                  <button
                    onClick={() =>
                      handleSend(
                        "Diabetes Type 2\nFatigue, frequent urination, increased thirst"
                      )
                    }
                    className="p-5 rounded-lg border border-[var(--primary-light)] bg-[var(--accent-light)] hover:bg-white transition-all text-left shadow-sm hover:shadow-md"
                  >
                    <div className="flex items-center mb-2">
                      <div className="w-7 h-7 rounded-full bg-[var(--primary)] text-white flex items-center justify-center mr-2">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="h-4 w-4"
                          viewBox="0 0 20 20"
                          fill="currentColor"
                        >
                          <path
                            fillRule="evenodd"
                            d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z"
                            clipRule="evenodd"
                          />
                        </svg>
                      </div>
                      <h3 className="font-semibold text-[var(--primary-dark)]">
                        Diabetes Management
                      </h3>
                    </div>
                    <p className="text-sm text-[var(--neutral-content)] border-l-2 border-[var(--secondary)] pl-3 ml-1">
                      &quot;Diabetes Type 2 with fatigue and thirst&quot;
                    </p>
                  </button>
                  <button
                    onClick={() =>
                      handleSend(
                        "High Blood Pressure\nHeadaches, dizziness, shortness of breath"
                      )
                    }
                    className="p-5 rounded-lg border border-[var(--primary-light)] bg-[var(--accent-light)] hover:bg-white transition-all text-left shadow-sm hover:shadow-md"
                  >
                    <div className="flex items-center mb-2">
                      <div className="w-7 h-7 rounded-full bg-[var(--primary)] text-white flex items-center justify-center mr-2">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="h-4 w-4"
                          viewBox="0 0 20 20"
                          fill="currentColor"
                        >
                          <path
                            fillRule="evenodd"
                            d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z"
                            clipRule="evenodd"
                          />
                        </svg>
                      </div>
                      <h3 className="font-semibold text-[var(--primary-dark)]">
                        Heart Health
                      </h3>
                    </div>
                    <p className="text-sm text-[var(--neutral-content)] border-l-2 border-[var(--secondary)] pl-3 ml-1">
                      &quot;High Blood Pressure with headaches&quot;
                    </p>
                  </button>
                </div>
              </div>
            </div>
          )}

          {messages.length > 0 && (
            <div className="pb-24 pt-5 mx-auto max-w-3xl px-4">
              {messages.map((m, i) => (
                <MessageBubble key={i} sender={m.sender} text={m.text} />
              ))}
              {loading && <MessageBubble sender="ai" text="Thinking..." />}
            </div>
          )}
        </div>

        {/* Input fixed at bottom */}
        <div className="fixed bottom-0 left-0 right-0 md:pl-64">
          <div className="mx-auto max-w-3xl px-4 pb-6">
            <InputBox onSend={handleSend} disabled={loading} />
          </div>
        </div>
      </div>
    </div>
  );
}
