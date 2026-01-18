import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { Message } from "../lib/chatTypes";

interface MessageBubbleProps extends Pick<Message, "sender" | "text"> {
  disease?: string;
  symptoms?: string;
  age?: number;
  gender?: string;
}

export default function MessageBubble({ sender, text, disease, symptoms, age, gender }: MessageBubbleProps) {
  const isUser = sender === "user";

  // Fallback: parse disease/symptoms from text if structured fields not present
  const parsedLines = text ? text.split('\n').map((l) => l.trim()).filter(Boolean) : [];
  const diseaseText = disease || parsedLines[0] || "";
  const symptomsText = symptoms || (parsedLines.length > 1 ? parsedLines.slice(1).join(', ') : '');

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
            <div className="relative">
              <div className="whitespace-pre-wrap text-[var(--foreground)] bg-[var(--accent-light)] p-4 rounded-lg border-l-4 border-[var(--primary)] shadow-sm">
                {diseaseText && <div className="font-semibold mb-2">{diseaseText}</div>}
                {symptomsText && <div className="text-sm text-[var(--primary-dark)]">{symptomsText}</div>}
              </div>

              {/* age/gender pill at top-right of the bubble */}
              {(age || gender) && (
                <div className="absolute top-0 right-0 translate-x-1 -translate-y-1">
                  <div className="flex items-center gap-2 bg-white border border-[var(--primary-light)] text-[var(--primary-dark)] rounded-full px-3 py-1 text-xs shadow-sm">
                    {age ? <span>{age}y</span> : null}
                    {age && gender ? <span className="opacity-50">•</span> : null}
                    {gender ? <span className="capitalize">{gender}</span> : null}
                  </div>
                </div>
              )}
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
