"use client";

import { useState } from "react";

interface InputBoxProps {
  onSend: (message: string) => void;
  disabled?: boolean;
}

export default function InputBox({ onSend, disabled }: InputBoxProps) {
  const [input, setInput] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim()) return;
    onSend(input.trim());
    setInput("");
  }

  return (
    <div className="relative">
      <form
        onSubmit={handleSubmit}
        className="relative flex items-stretch overflow-hidden border-2 border-[var(--primary-light)] rounded-md shadow-lg"
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Describe your wellness concerns..."
          rows={1}
          className="flex-1 px-4 py-3 resize-none border-none overflow-hidden rounded-l-md text-[var(--input-text)] bg-transparent placeholder:text-[var(--input-placeholder)] focus:outline-none font-medium m-0 block w-full"
          disabled={disabled}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              if (input.trim()) handleSubmit(e);
            }
          }}
        />
        <button
          type="submit"
          disabled={disabled || !input.trim()}
          className={`px-4 py-3 rounded-r-md flex-shrink-0 border-l border-[var(--primary-light)] h-auto flex items-center ${
            input.trim()
              ? "text-white bg-[var(--primary)] hover:bg-[var(--primary-dark)] font-medium"
              : "text-[var(--input-placeholder)] bg-[var(--neutral)] cursor-not-allowed"
          } transition-all`}
        >
          <div className="flex items-center gap-1">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="currentColor"
              className="w-5 h-5"
            >
              <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
            </svg>
            <span>Send</span>
          </div>
        </button>
      </form>

      <div className="text-center text-xs text-[var(--primary-dark)] font-medium mt-3 px-3 py-1 bg-[var(--neutral)] rounded-md inline-block mx-auto">
        Your Indian Thali provides personalized nutritional guidance for Indian
        Thali recommendations
      </div>
    </div>
  );
}
