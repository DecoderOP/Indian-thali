export interface Message {
  sender: "user" | "ai";
  text: string;
  // Optional structured metadata for user messages
  disease?: string;
  symptoms?: string; // comma-separated or free text
  age?: number;
  gender?: string;
}

export interface ConversationData {
  title: string;
  messages: Message[];
  timestamp: number;
}
