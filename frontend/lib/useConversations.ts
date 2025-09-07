"use client";

import { useState, useEffect } from "react";
import { Message, ConversationData } from "./chatTypes";

// Custom hook for managing conversations
export function useConversations() {
  const [conversations, setConversations] = useState<Record<string, ConversationData>>({});

  // Load conversations from localStorage on component mount
  useEffect(() => {
    try {
      const savedConversations = localStorage.getItem('thaliConversations');
      if (savedConversations) {
        setConversations(JSON.parse(savedConversations));
      }
    } catch (error) {
      console.error('Failed to load conversations from localStorage:', error);
    }
  }, []);

  // Helper function to find if a conversation with similar content exists
  const findSimilarConversation = (messages: Message[], activeId: string | null): string | null => {
    if (messages.length < 2) return null;
    
    // Get first user message as identifier
    const firstUserMsg = messages.find(m => m.sender === 'user')?.text;
    if (!firstUserMsg) return null;
    
    // Look for conversations with same first user message
    for (const [id, conv] of Object.entries(conversations)) {
      const convFirstUserMsg = conv.messages.find(m => m.sender === 'user')?.text;
      if (convFirstUserMsg === firstUserMsg && id !== activeId) {
        return id;
      }
    }
    return null;
  };
  
  // Save conversation
  const saveConversation = (messages: Message[], activeId: string | null = null) => {
    if (messages.length < 2) return; // Don't save empty conversations or with just one message

    // Create a title from the first user message
    const firstUserMessage = messages.find(m => m.sender === 'user');
    const title = firstUserMessage 
      ? firstUserMessage.text.substring(0, 30) + (firstUserMessage.text.length > 30 ? '...' : '')
      : 'New Conversation';

    // Use the provided ID, find similar conversation, or create a new ID
    const id = activeId || findSimilarConversation(messages, activeId) || `conv_${Date.now()}`;
    
    const updatedConversations = {
      ...conversations,
      [id]: {
        title,
        messages: [...messages],
        timestamp: Date.now()
      }
    };

    // Save to state and localStorage
    setConversations(updatedConversations);
    try {
      localStorage.setItem('thaliConversations', JSON.stringify(updatedConversations));
    } catch (error) {
      console.error('Failed to save conversations to localStorage:', error);
    }
  };

  // Delete conversation
  const deleteConversation = (id: string) => {
    const { [id]: removed, ...remainingConversations } = conversations;
    
    setConversations(remainingConversations);
    try {
      localStorage.setItem('thaliConversations', JSON.stringify(remainingConversations));
    } catch (error) {
      console.error('Failed to update localStorage after deletion:', error);
    }
  };

  // Clear all conversations
  const clearAllConversations = () => {
    setConversations({});
    try {
      localStorage.removeItem('thaliConversations');
    } catch (error) {
      console.error('Failed to clear localStorage:', error);
    }
  };

  return {
    conversations,
    saveConversation,
    deleteConversation,
    clearAllConversations
  };
}
