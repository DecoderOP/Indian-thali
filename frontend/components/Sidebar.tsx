"use client";

import React from "react";
import { Message } from "../lib/chatTypes";

interface SidebarProps {
  onNewConsultation: () => void;
  onLoadConversation: (conversationId: string) => void;
  onDeleteConversation: (conversationId: string) => void;
  conversations: Record<string, { title: string; messages: Message[] }>;
  activeConversationId?: string | null;
  onResize?: (width: number) => void;
}

export default function Sidebar({
  onNewConsultation,
  onLoadConversation,
  onDeleteConversation,
  conversations,
  activeConversationId,
  onResize,
}: SidebarProps) {
  const [width, setWidth] = React.useState(256); // 256px = 16rem (64 in Tailwind)
  const [isResizing, setIsResizing] = React.useState(false);
  const [isMobileOpen, setIsMobileOpen] = React.useState(false);
  const minWidth = 180; // Minimum sidebar width
  const maxWidth = 480; // Maximum sidebar width
  const sidebarRef = React.useRef<HTMLDivElement>(null);

  // Handle mouse down on the resize handle
  const startResizing = React.useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  }, []);

  // Handle mouse move for resizing
  React.useEffect(() => {
    const handleResize = (e: MouseEvent) => {
      if (isResizing) {
        const newWidth = e.clientX;
        if (newWidth >= minWidth && newWidth <= maxWidth) {
          setWidth(newWidth);
        }
      }
    };

    const stopResizing = () => {
      setIsResizing(false);
    };

    document.addEventListener("mousemove", handleResize);
    document.addEventListener("mouseup", stopResizing);
    return () => {
      document.removeEventListener("mousemove", handleResize);
      document.removeEventListener("mouseup", stopResizing);
    };
  }, [isResizing]);

  // Toggle sidebar for mobile view
  const toggleSidebar = () => {
    setIsMobileOpen(!isMobileOpen);
  };

  // Notify parent component of width changes
  React.useEffect(() => {
    if (onResize) {
      onResize(isMobileOpen ? width : 0);
    }
    // Intentionally excluding onResize from dependencies to prevent infinite loops
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, isMobileOpen]);

  // Save sidebar width preference to localStorage
  React.useEffect(() => {
    // Only save if not on mobile and it's not the default width
    if (!isResizing && width !== 256) {
      try {
        localStorage.setItem("thaliSidebarWidth", width.toString());
      } catch (error) {
        console.error("Failed to save sidebar width preference:", error);
      }
    }
  }, [isResizing, width]);

  // Load saved width preference
  React.useEffect(() => {
    try {
      const savedWidth = localStorage.getItem("thaliSidebarWidth");
      if (savedWidth) {
        const parsedWidth = parseInt(savedWidth, 10);
        if (parsedWidth >= minWidth && parsedWidth <= maxWidth) {
          setWidth(parsedWidth);
        }
      }
    } catch (error) {
      console.error("Failed to load sidebar width preference:", error);
    }
  }, []);

  return (
    <div className={isResizing ? "cursor-ew-resize" : ""}>
      {/* Mobile Toggle Button */}
      <button
        onClick={toggleSidebar}
        className="md:hidden fixed top-4 left-4 z-50 p-2 rounded-md bg-[var(--primary)] text-white shadow-lg"
        aria-label="Toggle sidebar"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-6 w-6"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 6h16M4 12h16M4 18h16"
          />
        </svg>
      </button>

      {/* Sidebar */}
      <div
        className={`${
          isMobileOpen ? "fixed inset-0 z-40 flex" : "hidden"
        } md:flex md:flex-col md:fixed md:inset-y-0 bg-[var(--sidebar-bg)] border-r border-[var(--border-color)] transition-all duration-300 ease-in-out`}
        ref={sidebarRef}
        style={{ width: `${width}px` }}
      >
        <div className="flex flex-col h-full relative">
          <div className="flex items-center justify-between px-4 py-3 bg-[var(--primary)] text-white mb-4">
            <h2 className="font-semibold text-lg">Indian Thali</h2>
            {/* Mobile close button */}
            <button
              className="md:hidden text-white hover:text-gray-200"
              onClick={() => setIsMobileOpen(false)}
              aria-label="Close sidebar"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-5 w-5"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                  clipRule="evenodd"
                />
              </svg>
            </button>
          </div>

          <div className="px-3 space-y-2 mb-6">
            <button
              onClick={onNewConsultation}
              className="w-full flex items-center gap-3 rounded-md py-3 px-4 text-sm bg-[var(--accent-light)] border border-[var(--accent)] text-[var(--primary-dark)] font-medium transition-all hover:bg-[var(--secondary)] hover:border-[var(--primary)]"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-5 w-5"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M10 3a1 1 0 00-1 1v5H4a1 1 0 100 2h5v5a1 1 0 102 0v-5h5a1 1 0 100-2h-5V4a1 1 0 00-1-1z"
                  clipRule="evenodd"
                />
              </svg>
              New Wellness Consultation
            </button>
          </div>

          <div className="flex-1 overflow-y-auto px-3">
            {Object.keys(conversations).length > 0 && (
              <>
                <div className="flex items-center gap-2 px-1 mb-3">
                  <div className="h-0.5 flex-1 bg-[var(--primary-light)] opacity-30"></div>
                  <div className="text-xs font-semibold text-[var(--primary)] uppercase tracking-wider px-2">
                    Previous Consultations
                  </div>
                  <div className="h-0.5 flex-1 bg-[var(--primary-light)] opacity-30"></div>
                </div>
                <div className="space-y-2">
                  {Object.entries(conversations).map(
                    ([id, { title, messages }]) => (
                      <div key={id} className="flex items-center group">
                        <button
                          onClick={() => onLoadConversation(id)}
                          className={`flex-grow text-left py-2.5 px-3 rounded-md text-sm border-l-2 ${
                            activeConversationId === id
                              ? "border-[var(--primary)] bg-[var(--neutral)]"
                              : "border-transparent"
                          } hover:border-[var(--primary)] hover:bg-[var(--neutral)] text-[var(--neutral-content)] transition-all`}
                        >
                          <div className="truncate font-medium text-[var(--primary-dark)]">
                            {title}
                          </div>
                          <div className="text-xs text-[var(--neutral-content)] mt-0.5 truncate">
                            {messages[0]?.text.substring(0, 35)}...
                          </div>
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteConversation(id);
                          }}
                          className="p-1 mr-1 text-[var(--neutral-content)] hover:text-red-500 hover:bg-[var(--neutral)] rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                          title="Delete conversation"
                        >
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-4 w-4"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                            />
                          </svg>
                        </button>
                      </div>
                    )
                  )}
                </div>
              </>
            )}
          </div>
        </div>

        {/* Resize Handle */}
        <div
          className={`absolute top-0 right-0 w-1.5 h-full cursor-ew-resize transition-all ${
            isResizing
              ? "bg-[var(--primary)] opacity-50"
              : "hover:bg-[var(--primary)] opacity-0 hover:opacity-40"
          } md:block hidden`}
          onMouseDown={startResizing}
          title="Resize sidebar"
        />
      </div>

      {/* Overlay for mobile */}
      {isMobileOpen && (
        <div
          className="md:hidden fixed inset-0 bg-black bg-opacity-50 z-30"
          onClick={() => setIsMobileOpen(false)}
        />
      )}
    </div>
  );
}
