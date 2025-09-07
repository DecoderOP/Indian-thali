export interface Message {
  sender: "user" | "ai";
  text: string;
}

export interface ConversationData {
  title: string;
  messages: Message[];
  timestamp: number;
}
