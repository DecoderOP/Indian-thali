import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { Message } from "../lib/chatTypes";

interface MessageBubbleProps extends Pick<Message, "sender" | "text"> {}

export default function MessageBubble({ sender, text }: MessageBubbleProps) {
  const isUser = sender === "user";

  return (
    <div
      className={`w-full py-6 ${isUser ? "bg-[var(--neutral)]" : "bg-[var(--background)]"} border-b border-[var(--border-color)]`}
    >
      <div className="max-w-3xl mx-auto px-4 flex items-start gap-4">
        {/* Avatar */}
        <div
          className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${
            isUser
              ? "bg-[var(--secondary)] border-2 border-[var(--primary)]"
              : "bg-[var(--primary)] border-2 border-[var(--secondary)]"
          }`}
        >
          {isUser ? (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
          )}
        </div>

        {/* Message content */}
        <div className="prose prose-sm max-w-none flex-1">
          {isUser ? (
            <div className="whitespace-pre-wrap text-[var(--foreground)] bg-[var(--accent-light)] p-4 rounded-lg border-l-4 border-[var(--primary)] shadow-sm">
              {text}
            </div>
          ) : (
            <div className="text-[var(--foreground)] bg-white p-4 rounded-lg border border-[var(--border-color)] shadow-sm">
              <div className="text-[var(--primary)] font-semibold mb-2">Wellness Recommendation:</div>
              <div className="prose">
                <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>{text}</ReactMarkdown>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
