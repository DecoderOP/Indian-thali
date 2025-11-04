"use client";

import { useState } from "react";

interface InputBoxProps {
  onSend: (payload: {
    disease: string;
    symptoms: string;
    age: number;
    gender: string;
  }) => void;
  disabled?: boolean;
}

export default function InputBox({ onSend, disabled }: InputBoxProps) {
  const [query, setQuery] = useState("");
  const [age, setAge] = useState<number | "">("");
  const [gender, setGender] = useState("male");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;

    // For the pill-style input we treat the single-line query as the main concern
    onSend({
      disease: query.trim(),
      symptoms: "",
      age: Number(age) || 30,
      gender,
    });

    setQuery("");
    setAge("");
  }

  return (
    <div className="w-full">
      {/* Age & Gender row above the pill input */}
      <form onSubmit={handleSubmit} className="w-full">
        <div className="flex items-center gap-3 mb-3">
          <input
            type="number"
            min={5}
            max={120}
            value={age}
            onChange={(e) => setAge(e.target.value ? Number(e.target.value) : "")}
            placeholder="Age"
            className="w-24 px-3 py-2 rounded-md border border-[var(--primary-light)] text-[var(--input-text)] focus:ring-2 focus:ring-[var(--primary)] focus:outline-none bg-white"
            disabled={disabled}
            aria-label="Age"
          />

          <select
            value={gender}
            onChange={(e) => setGender(e.target.value)}
            className="px-3 py-2 rounded-md border border-[var(--primary-light)] text-[var(--input-text)] focus:ring-2 focus:ring-[var(--primary)] focus:outline-none bg-white"
            disabled={disabled}
            aria-label="Gender"
          >
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="other">Other</option>
          </select>
        </div>

        {/* Pill-style input */}
        <div className="flex items-center">
          <label htmlFor="chat-query" className="sr-only">Ask anything</label>
          <div className="flex items-center w-full bg-white shadow-sm rounded-sm border border-[var(--primary-light)] px-4 py-2">
            <input
              id="chat-query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask anything"
              className="flex-1 bg-transparent placeholder:text-[var(--input-placeholder)] text-[var(--input-text)] outline-none px-2 py-1"
              disabled={disabled}
              aria-label="Ask anything"
            />

            <button
              type="submit"
              aria-label="Send"
              disabled={disabled || !query.trim()}
              className={`ml-3 flex items-center justify-center w-10 h-10 rounded-full transition-colors ${
                query.trim()
                  ? 'bg-[var(--primary)] text-white hover:bg-[var(--primary-dark)]'
                  : 'bg-[var(--neutral)] text-[var(--input-placeholder)] cursor-not-allowed'
              }`}
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" className="w-5 h-5" fill="currentColor">
                <path d="M2 21l21-9L2 3v7l15 2-15 2v7z" />
              </svg>
            </button>
          </div>
        </div>
      </form>

      <p className="text-center text-xs text-[var(--primary-dark)] font-medium mt-3 px-3 py-1 bg-[var(--neutral)] rounded-md inline-block mx-auto">
        Your Indian Thali provides hyper-personalized, evidence-based nutrition guidance.
      </p>
    </div>
  );
}
