"use client";
import React, { useState, useEffect, useRef } from 'react';
import { 
  Mic, Send, Volume2, HeartPulse, Scale, 
  ShieldCheck, History, Menu, X, PlusCircle, 
  ExternalLink, FileText, Search, Loader2, Info, Trash2
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

type EvidenceItem = {
  citation_tag: string;
  source_name: string;
  source_url: string;
  original_excerpt: string;
  rerank_score?: number | null;
  similarity?: number | null;
};

type SourceItem = {
  title?: string;
  content?: string;
  summary?: string;
  source_url?: string;
  url?: string;
  rerank_score?: number | null;
  similarity?: number | null;
  chunk_index?: number | null;
  total_chunks?: number | null;
  page_number?: number | null;
  page_start?: number | null;
  page_end?: number | null;
  section?: string | null;
  subsection?: string | null;
  source_type?: string | null;
};

type PreviewChunk = {
  title?: string;
  content: string;
  source_url: string;
  chunk_index?: number | null;
  total_chunks?: number | null;
  page_number?: number | null;
  page_start?: number | null;
  page_end?: number | null;
  section?: string | null;
  subsection?: string | null;
  source_type?: string | null;
};

type SourcePreviewPayload = {
  source_url: string;
  source_title?: string;
  source_type?: string;
  chunk_count: number;
  highlight_chunk_index?: number | null;
  highlight_chunk_position?: number | null;
  chunks: PreviewChunk[];
  reconstructed_markdown?: string;
};

type ChatMessage = {
  role: string;
  text: string;
  source?: string;
  sources?: any[];
  evidence?: EvidenceItem[];
  intent?: string;
  ragUsed?: boolean;
  summaryMode?: boolean;
  summaryModeReason?: string;
  dialect?: string;
  status?: string;
  detectedLanguage?: string;
  debugLogs?: string[];
  originalText?: string;
  isSimplifying?: boolean;
};

type ChatHistoryItem = {
  role: 'user' | 'assistant';
  text: string;
};

type ConversationRecord = {
  id: string;
  title: string;
  summary: string;
  created_at: string;
  updated_at: string;
  messages?: Array<{
    role: string;
    text: string;
    created_at?: string;
  }>;
};

type BackendHealthPayload = {
  status?: string;
  backend?: {
    rag_initialized?: boolean;
    warmup_status?: string;
    warmup_error?: string | null;
    message?: string;
  };
};

const MAX_HISTORY_MESSAGES = 6;
const CHAT_HISTORY_STORAGE_KEY = 'chat_history';
const CONVERSATION_ID_STORAGE_KEY = 'current_conversation_id';

const getWelcomeMessage = (): ChatMessage => ({
  role: 'assistant',
  text: 'Selamat Datang! I am your Public Service AI. I can explain government policies in simple language or local dialects.',
  dialect: 'Standard Malay',
});

const formatConversationTime = (isoDate: string): string => {
  const timestamp = new Date(isoDate).getTime();
  if (Number.isNaN(timestamp)) {
    return 'Recently';
  }

  const minutes = Math.max(1, Math.floor((Date.now() - timestamp) / 60000));
  if (minutes < 60) {
    return `${minutes} min ago`;
  }
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    return `${hours} hour${hours === 1 ? '' : 's'} ago`;
  }
  const days = Math.floor(hours / 24);
  return `${days} day${days === 1 ? '' : 's'} ago`;
};

export default function InclusiveApp() {
  const [input, setInput] = useState("");
  const [language, setLanguage] = useState("ms-MY");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isListening, setIsListening] = useState(false);
  const [speakingIndex, setSpeakingIndex] = useState<number | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [showDebugPanel, setShowDebugPanel] = useState(false);
  const [currentDebugLogs, setCurrentDebugLogs] = useState<string[]>([]);
  const [warmupStatus, setWarmupStatus] = useState<'checking' | 'warming' | 'ready' | 'failed'>('checking');
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<ConversationRecord[]>([]);
  const [expandedEvidence, setExpandedEvidence] = useState<Set<number>>(new Set());
  const scrollRef = useRef<HTMLDivElement>(null);
  const [previewData, setPreviewData] = useState<SourcePreviewPayload | null>(null);
  const [previewOwnerMessageIndex, setPreviewOwnerMessageIndex] = useState<number | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const previewChunkRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [currentAudio, setCurrentAudio] = useState<HTMLAudioElement | null>(null);
  const [showStarterChips, setShowStarterChips] = useState(false);
  const showPreviewPanel = previewLoading || !!previewError || !!previewData;

  const syncLocalMessages = (nextMessages: ChatMessage[]) => {
    setMessages(nextMessages);
    localStorage.setItem(CHAT_HISTORY_STORAGE_KEY, JSON.stringify(nextMessages));
  };

  const stopAudio = () => {
    setCurrentAudio((prevAudio) => {
      if (prevAudio) {
        prevAudio.pause();
      }
      return null;
    });
    setSpeakingIndex(null);
    window.speechSynthesis.cancel();
  };

  const resetPreview = () => {
    setPreviewData(null);
    setPreviewOwnerMessageIndex(null);
    setPreviewLoading(false);
    setPreviewError(null);
    previewChunkRefs.current = {};
  };

  const openReconstructedPreview = async (source: SourceItem, messageIndex: number) => {
    const sourceUrl = source.source_url || source.url;
    if (!sourceUrl) return;

    setPreviewLoading(true);
    setPreviewError(null);
    setPreviewOwnerMessageIndex(messageIndex);

    try {
      const params = new URLSearchParams({ source_url: sourceUrl });
      if (typeof source.chunk_index === 'number') {
        params.set('highlight_chunk_index', String(source.chunk_index));
      }
      if (source.title) {
        params.set('highlight_title', source.title);
      }

      const response = await fetch(`/api/source-preview?${params.toString()}`, { cache: 'no-store' });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result?.error || 'Failed to load reconstructed preview');
      }

      setPreviewData(result as SourcePreviewPayload);
    } catch (error) {
      setPreviewData(null);
      setPreviewError(error instanceof Error ? error.message : 'Failed to load reconstructed preview');
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleCitationJump = async (
    messageIndex: number,
    citationTag: string,
    sources?: SourceItem[]
  ) => {
    if (!sources || sources.length === 0) return;

    const match = citationTag.match(/^S(\d+)$/i);
    if (!match) return;

    const sourceIdx = Number(match[1]) - 1;
    if (Number.isNaN(sourceIdx) || sourceIdx < 0 || sourceIdx >= sources.length) return;

    await openReconstructedPreview(sources[sourceIdx], messageIndex);
  };

  const refreshConversations = async () => {
    try {
      const response = await fetch('/api/conversations', { cache: 'no-store' });
      const result = await response.json();
      if (!response.ok) {
        console.warn('Failed to load conversations:', result?.error || response.statusText);
        setConversations([]);
        return [];
      }
      setConversations(result.conversations || []);
      return result.conversations || [];
    } catch (error) {
      console.warn('Failed to load conversations:', error);
      setConversations([]);
      return [];
    }
  };

  const loadConversation = async (conversationId: string) => {
    stopAudio();
    setShowStarterChips(false);
    
    const response = await fetch(`/api/conversations/${conversationId}`, { cache: 'no-store' });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || 'Failed to load conversation');
    }

    const nextMessages = result.messages && result.messages.length > 0
      ? result.messages.map((message: any) => ({
          role: message.role,
          text: message.text,
          sources: message.sources || [],
          evidence: message.evidence || [],
          detectedLanguage: message.detectedLanguage,
          intent: message.intent,
          ragUsed: message.ragUsed,
          summaryMode: message.summaryMode,
          summaryModeReason: message.summaryModeReason,
          status: message.status,
          debugLogs: message.debugLogs || [],
          created_at: message.created_at,
        }))
      : [getWelcomeMessage()];

    setCurrentConversationId(result.id);
    localStorage.setItem(CONVERSATION_ID_STORAGE_KEY, result.id);
    syncLocalMessages(nextMessages);
    setCurrentDebugLogs([]);
    resetPreview();
    return result as ConversationRecord;
  };

  const createConversation = async () => {
    stopAudio();
    
    try {
      const response = await fetch('/api/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.error || 'Failed to create conversation');
      }

      setCurrentConversationId(result.id);
      localStorage.setItem(CONVERSATION_ID_STORAGE_KEY, result.id);
      syncLocalMessages([getWelcomeMessage()]);
      setShowStarterChips(true);
      setCurrentDebugLogs([]);
      resetPreview();
      await refreshConversations();
      return result as ConversationRecord;
    } catch (error) {
      console.warn('Failed to create conversation, falling back to local chat:', error);
      setCurrentConversationId(null);
      localStorage.removeItem(CONVERSATION_ID_STORAGE_KEY);
      syncLocalMessages([getWelcomeMessage()]);
      setShowStarterChips(true);
      setCurrentDebugLogs([]);
      resetPreview();
      return null;
    }
  };

  const deleteConversation = async (conversationId: string) => {
    const confirmed = window.confirm('Delete this conversation permanently?');
    if (!confirmed) return;

    try {
      const response = await fetch(`/api/conversations/${conversationId}`, {
        method: 'DELETE',
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.error || 'Failed to delete conversation');
      }

      const updatedConversations = await refreshConversations();
      if (currentConversationId === conversationId) {
        if (updatedConversations.length > 0) {
          await loadConversation(updatedConversations[0].id);
        } else {
          await createConversation();
        }
      }
    } catch (error) {
      console.warn('Failed to delete conversation:', error);
    }
  };

  // Auto-scroll to bottom on new message
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  useEffect(() => {
    if (!previewData) return;

    const hasIndex = previewData.highlight_chunk_index != null;
    const hasPosition = previewData.highlight_chunk_position != null;
    if (!hasIndex && !hasPosition) return;

    const timer = setTimeout(() => {
      const targetKey = hasIndex
        ? `idx:${previewData.highlight_chunk_index as number}`
        : `pos:${previewData.highlight_chunk_position as number}`;
      const node = previewChunkRefs.current[targetKey];
      if (node) {
        node.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }, 120);

    return () => clearTimeout(timer);
  }, [previewData]);

  useEffect(() => {
    let isMounted = true;

    const bootstrapConversation = async () => {
      // Show existing local history immediately while server sync happens.
      const savedMessages = localStorage.getItem(CHAT_HISTORY_STORAGE_KEY);
      if (savedMessages) {
        try {
          const parsedMessages = JSON.parse(savedMessages);
          if (Array.isArray(parsedMessages) && parsedMessages.length > 0) {
            setMessages(parsedMessages);
          }
        } catch (error) {
          console.warn('Failed to parse saved chat history:', error);
        }
      } else {
        setMessages([getWelcomeMessage()]);
      }

      const savedConversationId = localStorage.getItem(CONVERSATION_ID_STORAGE_KEY);

      try {
        const availableConversations = await refreshConversations();
        if (!isMounted) return;

        if (savedConversationId) {
          const matchingConversation = availableConversations.find((conversation: ConversationRecord) => conversation.id === savedConversationId);
          if (matchingConversation) {
            await loadConversation(savedConversationId);
            return;
          }
        }

        if (availableConversations.length > 0) {
          await loadConversation(availableConversations[0].id);
          return;
        }

        await createConversation();
      } catch (error) {
        console.warn('Conversation bootstrap fallback activated:', error);
        // Keep already hydrated local history shown to user.
      }
    };

    bootstrapConversation();

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    let isMounted = true;
    let pollId: ReturnType<typeof setInterval> | null = null;

    const checkBackendWarmup = async () => {
      try {
        const response = await fetch('/api/chat', { method: 'GET', cache: 'no-store' });
        const result: BackendHealthPayload = await response.json();

        if (!isMounted) return;

        const backend = result.backend;
        const status = backend?.warmup_status;
        const ragReady = backend?.rag_initialized === true;

        if (ragReady || status === 'ready') {
          setWarmupStatus('ready');
          if (pollId) {
            clearInterval(pollId);
            pollId = null;
          }
          return;
        }

        if (status === 'failed') {
          setWarmupStatus('failed');
          return;
        }

        setWarmupStatus('warming');
      } catch {
        if (!isMounted) return;
        setWarmupStatus('checking');
      }
    };

    checkBackendWarmup();
    pollId = setInterval(checkBackendWarmup, 3000);

    return () => {
      isMounted = false;
      if (pollId) {
        clearInterval(pollId);
      }
    };
  }, []);

  const toggleListening = async () => {
    // If already listening, stop it manually
    if (isListening && mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsListening(false);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      const audioChunks: Blob[] = [];

      mediaRecorder.ondataavailable = (event) => audioChunks.push(event.data);

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        setIsTyping(false);
        setIsTranscribing(true); // Start transcription loading

        const formData = new FormData();
        formData.append('file', audioBlob, 'input_audio.wav');

        try {
          const response = await fetch('http://localhost:8000/transcribe', {
            method: 'POST',
            body: formData,
          });
          const data = await response.json();
          if (data.success) setInput(data.text);
        } catch (err) {
          console.error("Transcription failed:", err);
        } finally {
          setIsTranscribing(false);
          stream.getTracks().forEach(track => track.stop());
        }
      };

      mediaRecorder.start();
      setIsListening(true);
    } catch (err) {
      alert("Please allow microphone access.");
    }
  };

  const handleSimplify = async (messageIndex: number, text: string) => {
    // Set loading state for this message
    const newMessages = messages.map((m, i) => {
        if (i === messageIndex) {
            return { ...m, isSimplifying: true };
        }
        return m;
    });
    setMessages(newMessages);

    try {
        const response = await fetch('/api/simplify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });
        const result = await response.json();

        if (result.success) {
            const updatedMessages = messages.map((m, i) => {
                if (i === messageIndex) {
                    return {
                        ...m,
                        text: result.simplified_text,
                        originalText: m.text, // Store original text
                        isSimplifying: false,
                    };
                }
                return m;
            });
            setMessages(updatedMessages);
        } else {
            // Handle error - maybe show a toast notification
            console.error("Simplification failed:", result.error);
            // Revert loading state
            const revertedMessages = messages.map((m, i) => {
                if (i === messageIndex) {
                    return { ...m, isSimplifying: false };
                }
                return m;
            });
            setMessages(revertedMessages);
        }
    } catch (error) {
        console.error("Simplification failed:", error);
        // Revert loading state
        const revertedMessages = messages.map((m, i) => {
            if (i === messageIndex) {
                return { ...m, isSimplifying: false };
            }
            return m;
        });
        setMessages(revertedMessages);
    }
  };

  const handleSend = async (overrideInput?: string) => {
    const textToSend = overrideInput || input;
    if (!textToSend) return;
    setShowStarterChips(false);

    // Add user message to chat
    const newMessages = [...messages, { role: 'user', text: textToSend }];
    syncLocalMessages(newMessages);
    setInput("");
    setIsTyping(true);

    const conversationHistory: ChatHistoryItem[] = newMessages
      .filter((message): message is ChatMessage & { role: 'user' | 'assistant' } =>
        (message.role === 'user' || message.role === 'assistant') && !!message.text?.trim()
      )
      .slice(-MAX_HISTORY_MESSAGES)
      .map((message) => ({
        role: message.role,
        text: message.text.trim(),
      }));
    
    try {
      let activeConversationId = currentConversationId;
      if (!activeConversationId) {
        const createdConversation = await createConversation();
        if (createdConversation) {
          activeConversationId = createdConversation.id;
        }
      }

      // Call Python RAG backend via Next.js API route
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query: textToSend,
          top_k: 5,
          conversation_id: activeConversationId,
          conversation_history: conversationHistory,
        }),
      });

      const result = await response.json();

      setIsTyping(false);

      if (result.answer) {
        // Success or no results (both have answer text)
        // Keep preview user-driven (click source card) instead of auto-opening.

        // Detect language name from code
        const languageMap: Record<string, string> = {
          'en': 'English',
          'ms': 'Malay (Bahasa Malaysia)',
          'zh': 'Chinese',
          'ta': 'Tamil',
          'th': 'Thai',
          'vi': 'Vietnamese',
          'tl': 'Tagalog',
          'id': 'Indonesian',
        };

        const detectedLanguage = languageMap[result.detected_language] || result.detected_language;

        const assistantMessage = { 
          role: 'assistant', 
          text: result.answer,
          source: undefined,
          sources: result.sources, // Store all sources
          evidence: result.evidence || [],
          intent: result.intent,
          ragUsed: result.rag_used,
          summaryMode: !!result.summary_mode,
          summaryModeReason: result.summary_mode_reason,
          dialect: detectedLanguage,
          detectedLanguage: result.detected_language,
          status: result.success ? 'verified' : 'no_results',
          debugLogs: result.debug_logs || []
        };

        const updatedMessages = [...newMessages, assistantMessage];
        syncLocalMessages(updatedMessages);
        if (result.conversation_id) {
          setCurrentConversationId(result.conversation_id);
          localStorage.setItem(CONVERSATION_ID_STORAGE_KEY, result.conversation_id);
        }
        
        // Update debug logs display
        if (result.debug_logs && result.debug_logs.length > 0) {
          setCurrentDebugLogs(result.debug_logs);
        }
        await refreshConversations();
      } else if (result.error) {
        // Actual error with error message
        syncLocalMessages([...newMessages, { 
          role: 'assistant', 
          text: result.error,
          dialect: "Error"
        }]);
      } else {
        // Unknown error
        syncLocalMessages([...newMessages, { 
          role: 'assistant', 
          text: "Sorry, I couldn't process your request. Please try again.",
          dialect: "Error"
        }]);
      }
    } catch (error) {
      setIsTyping(false);
      console.error('Chat error:', error);
      
      syncLocalMessages([...newMessages, { 
        role: 'assistant', 
        text: "I'm having trouble connecting to the backend. Please ensure both servers are running:\n\n• Python backend: http://localhost:8000\n• Next.js frontend: http://localhost:3001",
        dialect: "Connection Error"
      }]);
    }
  };

  return (
    <div className="flex h-[100dvh] bg-[#F8FAFC] font-sans text-slate-900 antialiased">
      
      {/* SIDEBAR OVERLAY */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm z-40 lg:hidden" 
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* SIDEBAR */}
      <aside className={`fixed inset-y-0 left-0 z-50 w-80 bg-slate-100 border-r border-slate-200 transform transition-transform duration-300 ease-in-out lg:relative lg:translate-x-0 ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="p-6 flex flex-col h-full">
          <div className="flex justify-end items-center mb-8">
            <button onClick={() => setIsSidebarOpen(false)} className="lg:hidden p-2 text-slate-400 hover:bg-slate-100 rounded-full transition"><X /></button>
          </div>
          
          <button 
            onClick={async () => {
              await createConversation();
              setIsSidebarOpen(false);
            }} 
            className="flex items-center justify-center gap-2 w-full py-3 px-4 mb-6 rounded-xl bg-slate-900 text-white font-semibold hover:bg-slate-800 transition-all shadow-sm active:scale-95"
          >
            <PlusCircle size={18}/> New Chat
          </button>

          <div className="flex-1 overflow-y-auto pr-2 -mr-2 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-slate-300 [&::-webkit-scrollbar-thumb]:rounded-full hover:[&::-webkit-scrollbar-thumb]:bg-slate-400">
            <p className="text-[11px] font-bold text-slate-400 uppercase tracking-[0.15em] mb-4">Recent Conversations</p>
            <div className="space-y-3">
              {conversations.map((conversation) => (
                <div key={conversation.id} className="relative group">
                  <button
                    onClick={async () => {
                      await loadConversation(conversation.id);
                      setIsSidebarOpen(false);
                    }}
                    className={`w-full p-4 pr-12 text-left rounded-xl border text-sm transition-colors ${
                      currentConversationId === conversation.id
                        ? 'bg-emerald-50 border-emerald-300 text-slate-700'
                        : 'bg-slate-50 border-slate-200 text-slate-500 hover:border-emerald-300'
                    }`}
                  >
                    <p className="font-semibold text-slate-700 mb-1 line-clamp-2">{conversation.title}</p>
                    <p className="text-[11px] opacity-60 mt-2">{formatConversationTime(conversation.updated_at)}</p>
                  </button>
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      deleteConversation(conversation.id);
                    }}
                    className="absolute top-3 right-3 p-1.5 rounded-lg bg-white border border-slate-200 text-slate-500 hover:text-rose-600 hover:border-rose-300 transition-opacity opacity-0 group-hover:opacity-100 focus:opacity-100"
                    title="Delete conversation"
                    aria-label="Delete conversation"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </aside>

      <main className={`flex-1 flex flex-col min-w-0 bg-white relative transition-all duration-300 ${showPreviewPanel ? 'lg:pr-[27rem]' : ''}`}>
        {/* HEADER */}
        <header className="sticky top-0 z-30 p-4 lg:px-8 bg-white/80 backdrop-blur-md border-b border-slate-100 flex justify-between items-center">
          <div className="flex items-center gap-4">
            <button onClick={() => setIsSidebarOpen(true)} className="lg:hidden p-2 text-slate-600 hover:bg-slate-100 rounded-xl"><Menu size={24} /></button>
            <div>
              <h1 className="font-extrabold text-slate-900 text-lg lg:text-xl">SuaraGov</h1>
              <p className="text-[10px] text-emerald-600 font-bold uppercase tracking-wider flex items-center gap-1">
                <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></span> Official Government AI
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            {warmupStatus !== 'ready' && (
              <div
                className={`px-3 py-1.5 rounded-full text-[11px] font-semibold border ${
                  warmupStatus === 'failed'
                    ? 'bg-rose-50 text-rose-700 border-rose-200'
                    : 'bg-amber-50 text-amber-700 border-amber-200'
                }`}
                title={
                  warmupStatus === 'failed'
                    ? 'Model warmup failed. First message may be slower while retrying.'
                    : 'Preparing AI models. First response may be slower.'
                }
              >
                {warmupStatus === 'failed' ? 'Model warmup failed' : 'Warming up model...'}
              </div>
            )}
            {/* Debug Panel Toggle */}
            <button 
              onClick={() => setShowDebugPanel(!showDebugPanel)}
              className={`px-3 py-2 rounded-xl text-xs font-bold transition-all border ${
                showDebugPanel 
                  ? 'bg-purple-100 text-purple-700 border-purple-300' 
                  : 'bg-slate-100 text-slate-600 border-slate-200 hover:bg-purple-50'
              }`}
              title="Toggle debug panel"
            >
              <Info size={16} />
            </button>
          </div>
        </header>

        {/* MESSAGES AREA */}
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto p-4 lg:p-10 space-y-8 scroll-smooth transition-all duration-300"
        >
          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2 duration-300`}>
              <div className={`relative max-w-[92%] sm:max-w-[85%] lg:max-w-[70%] px-4 py-3 sm:px-5 sm:py-4 lg:px-6 lg:py-4 rounded-[2rem] ${
                m.role === 'user' 
                ? 'bg-emerald-600 text-white rounded-br-none shadow-lg shadow-emerald-100' 
                : 'bg-slate-50 text-slate-800 border border-slate-100 rounded-bl-none'
              }`}>
                
                {m.role === 'assistant' && (
                  <div className="flex items-center gap-2 mb-3 flex-wrap">
                    {m.ragUsed === true && (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider bg-emerald-50 text-emerald-700 border border-emerald-200">
                        <Search size={10} /> RAG Verified
                      </span>
                    )}
                    {m.ragUsed === false && (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider bg-slate-100 text-slate-500 border border-slate-200">
                        💬 Direct Answer
                      </span>
                    )}
                    {m.summaryMode === true && (
                      <span
                        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider bg-amber-50 text-amber-700 border border-amber-200"
                        title={m.summaryModeReason ? `Summary mode: ${m.summaryModeReason}` : 'Summary mode enabled'}
                      >
                        📝 Summary Mode
                      </span>
                    )}
                  </div>
                )}

                <div className="prose prose-lg max-w-none text-base lg:text-lg leading-relaxed font-medium">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {m.text}
                  </ReactMarkdown>
                </div>

                {m.role === 'assistant' && (
                  <div className="mt-6 space-y-4">
                    {/* Evidence section — original source excerpts with citation tags */}
                    {m.evidence && m.evidence.length > 0 && (
                      <div className="space-y-1">
                        <button
                          type="button"
                          onClick={() => {
                            setExpandedEvidence(prev => {
                              const next = new Set(prev);
                              if (next.has(i)) next.delete(i); else next.add(i);
                              return next;
                            });
                          }}
                          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-emerald-50 text-emerald-800 border border-emerald-200 hover:bg-emerald-100 hover:text-emerald-900 transition-colors shadow-sm"
                        >
                          <FileText size={12} />
                          {expandedEvidence.has(i) ? '▾' : '▸'} Original Source Excerpts ({m.evidence.length})
                        </button>
                        {expandedEvidence.has(i) && (
                          <div className="space-y-2 mt-2">
                            {m.evidence.map((ev: EvidenceItem, evIdx: number) => (
                              <div key={evIdx} className="bg-white border border-emerald-100 rounded-xl p-3.5 shadow-sm transition-all hover:border-emerald-300 hover:shadow-md">
                                <div className="flex items-center justify-between mb-1.5">
                                  <button
                                    type="button"
                                    onClick={() => handleCitationJump(i, ev.citation_tag, m.sources as SourceItem[] | undefined)}
                                    className="text-[10px] font-bold text-emerald-800 bg-emerald-100 px-2 py-0.5 rounded-full hover:bg-emerald-200 hover:text-emerald-900 transition-colors"
                                    title="Jump to reconstructed source chunk"
                                  >
                                    [{ev.citation_tag}]
                                  </button>
                                  <span className="text-[10px] text-slate-500 font-medium truncate ml-2 flex-1 text-right">
                                    {ev.source_name}
                                    {(ev.rerank_score != null || ev.similarity != null) && (
                                      <span className="ml-1 text-emerald-600">
                                        · {(((ev.rerank_score ?? ev.similarity) as number) * 100).toFixed(0)}% selection
                                      </span>
                                    )}
                                  </span>
                                </div>
                                <p className="text-[11px] text-slate-600 italic leading-relaxed line-clamp-4">
                                  &ldquo;{ev.original_excerpt}&rdquo;
                                </p>
                                <button
                                  type="button"
                                  onClick={() => handleCitationJump(i, ev.citation_tag, m.sources as SourceItem[] | undefined)}
                                  className="mt-2 inline-flex items-center gap-1 text-[10px] font-semibold text-emerald-600 hover:text-emerald-800 transition-colors"
                                >
                                  <Search size={10} /> View in Document
                                </button>
                                {ev.source_url && (
                                  <button
                                    type="button"
                                    onClick={() => window.open(ev.source_url, '_blank')}
                                    className="mt-2 ml-4 inline-flex items-center gap-1 text-[10px] font-semibold text-slate-500 hover:text-emerald-600 transition-colors"
                                  >
                                    <ExternalLink size={10} /> View source
                                  </button>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Legacy single source support */}
                    {m.source && (!m.sources || m.sources.length === 0) && (
                      <div className="bg-white border border-emerald-100 rounded-2xl p-4 flex items-center justify-between group hover:border-emerald-300 hover:shadow-md transition-all cursor-pointer shadow-sm">
                        <div className="flex items-center gap-3 overflow-hidden">
                          <div className="p-2.5 bg-emerald-50 text-emerald-600 rounded-xl">
                            <FileText size={20} />
                          </div>
                          <div className="overflow-hidden">
                            <p className="text-xs font-bold text-slate-800">Verified Source Document</p>
                            <p className="text-[10px] text-slate-400 truncate w-full italic">Click to view official policy</p>
                          </div>
                        </div>
                        <ExternalLink size={18} className="text-slate-300 group-hover:text-emerald-500 transition-colors shrink-0" />
                      </div>
                    )}

                    <div className="flex flex-wrap gap-3">
                      <button 
                        onClick={() => {
                          if (speakingIndex === i) {
                            if (currentAudio) {
                              currentAudio.pause();
                              setCurrentAudio(null);
                            }
                            window.speechSynthesis.cancel();
                            setSpeakingIndex(null);
                          } else {
                            if (currentAudio) {
                              currentAudio.pause();
                            }
                            window.speechSynthesis.cancel(); // Clear queue
                            setSpeakingIndex(i);
                            
                            const ttsLang = m.detectedLanguage || language.split('-')[0] || 'en';
                            const url = `http://localhost:8000/api/tts?text=${encodeURIComponent(m.text)}&lang=${ttsLang}`;
                            
                            const audio = new Audio(url);
                            setCurrentAudio(audio);
                            
                            audio.onplay = () => setSpeakingIndex(i);
                            audio.onended = () => {
                              setSpeakingIndex(null);
                              setCurrentAudio(null);
                            };
                            audio.onerror = () => {
                              console.warn("Neural TTS failed, falling back to browser TTS.");
                              const utterance = new SpeechSynthesisUtterance(m.text);
                              utterance.lang = m.detectedLanguage || 'en';
                              utterance.onstart = () => setSpeakingIndex(i);
                              utterance.onend = () => { setSpeakingIndex(null); setCurrentAudio(null); };
                              utterance.onerror = () => { setSpeakingIndex(null); setCurrentAudio(null); };
                              window.speechSynthesis.speak(utterance);
                            };
                            
                            audio.play().catch(e => {
                              console.error("Audio playback blocked", e);
                              setSpeakingIndex(null);
                              setCurrentAudio(null);
                            });
                          }
                        }} 
                        className={`flex items-center gap-2 px-5 py-2.5 rounded-full text-sm font-bold transition-all shadow-sm border ${
                          speakingIndex === i 
                            ? 'bg-rose-50 border-rose-200 text-rose-700 hover:bg-rose-100' 
                            : 'bg-white border-slate-200 text-slate-700 hover:border-emerald-500 hover:text-emerald-700'
                        }`}
                      >
                        {speakingIndex === i ? (
                          <>
                            <div className="flex gap-1 items-center mr-1">
                              <span className="w-1 h-3 bg-rose-500 animate-pulse"></span>
                              <span className="w-1 h-3 bg-rose-500 animate-pulse delay-75"></span>
                            </div>
                            End
                          </>
                        ) : (
                          <>
                            <Volume2 size={18} /> Listen
                          </>
                        )}
                      </button>
                      <button 
                        onClick={() => {
                          if (m.originalText) {
                            // If already simplified, revert to original
                            const updatedMessages = messages.map((msg, index) => {
                              if (index === i) {
                                return { ...msg, text: msg.originalText ?? msg.text, originalText: undefined };
                              }
                              return msg;
                            });
                            setMessages(updatedMessages);
                          } else {
                            handleSimplify(i, m.text)
                          }
                        }}
                        disabled={m.isSimplifying}
                        className="flex items-center gap-2 bg-white border border-slate-200 text-slate-700 px-5 py-2.5 rounded-full text-sm font-bold hover:border-emerald-500 transition-all shadow-sm"
                      >
                        {m.isSimplifying ? (
                          <>
                            <Loader2 size={18} className="animate-spin" />
                            Simplifying...
                          </>
                        ) : (
                          <>
                            <PlusCircle size={18} />
                            {m.originalText ? 'Show Original' : 'Simpler Version'}
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}

          {isTyping && (
            <div className="flex justify-start">
              <div className="bg-slate-50 border border-slate-100 p-6 rounded-[2rem] rounded-bl-none flex gap-2">
                <span className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{animationDelay: '0ms'}}></span>
                <span className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{animationDelay: '150ms'}}></span>
                <span className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></span>
              </div>
            </div>
          )}
        </div>

        {/* INPUT AREA */}
        <footer className="p-4 lg:p-8 bg-white border-t border-slate-100 transition-all duration-300">
          {/* Floating Chips Area */}
          {showStarterChips && (
            <div className="max-w-4xl mx-auto mb-4">
              <div className="bg-transparent overflow-x-auto no-scrollbar pb-1">
                <div className="flex gap-2 w-max pr-1">
                  <button onClick={() => handleSend("Tell me about medical aid")} className="shrink-0 flex items-center gap-2 bg-white/80 backdrop-blur-sm border border-slate-200 px-4 py-2 rounded-xl text-sm font-semibold text-slate-600 hover:bg-emerald-50 hover:border-emerald-200 transition-colors shadow-sm"><HeartPulse size={16}/> Medical Aid</button>
                  <button onClick={() => handleSend("Welfare application")} className="shrink-0 flex items-center gap-2 bg-white/80 backdrop-blur-sm border border-slate-200 px-4 py-2 rounded-xl text-sm font-semibold text-slate-600 hover:bg-blue-50 hover:border-blue-200 transition-colors shadow-sm"><ShieldCheck size={16}/> Welfare</button>
                  <button onClick={() => handleSend("How to get legal help?")} className="shrink-0 flex items-center gap-2 bg-white/80 backdrop-blur-sm border border-slate-200 px-4 py-2 rounded-xl text-sm font-semibold text-slate-600 hover:bg-amber-50 hover:border-amber-200 transition-colors shadow-sm"><Scale size={16}/> Legal Help</button>
                </div>
              </div>
            </div>
          )}

          {/* Chat Input Area */}
          <div className="max-w-4xl mx-auto">
            <div className="relative flex items-end gap-3 bg-slate-50 border-2 border-slate-200 rounded-[2rem] p-2 focus-within:border-emerald-500 transition-all shadow-inner overflow-hidden min-h-[72px]">
              
              {/* LISTENING OVERLAY */}
              {isListening && (
                <div className="absolute inset-0 bg-rose-50/95 backdrop-blur-sm z-30 flex items-center px-6 animate-in fade-in zoom-in-95 duration-200">
                  <div className="flex gap-1.5 items-center mr-4">
                    <span className="w-1.5 h-4 bg-rose-500 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                    <span className="w-1.5 h-6 bg-rose-500 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                    <span className="w-1.5 h-4 bg-rose-500 rounded-full animate-bounce"></span>
                  </div>
                  <span className="text-sm font-bold text-rose-600 animate-pulse uppercase tracking-wider">
                    Listening... Tap mic to finish
                  </span>
                </div>
              )}

              {/* TRANSCRIBING OVERLAY */}
              {isTranscribing && (
                <div className="absolute inset-0 bg-white/90 backdrop-blur-md z-30 flex items-center justify-between px-6 animate-in slide-in-from-bottom-2 duration-300">
                  <div className="flex items-center">
                    <Loader2 size={18} className="animate-spin text-emerald-600 mr-3" />
                    <span className="text-sm font-bold text-slate-700 italic">Processing your voice...</span>
                  </div>
                  {/* Optional: Add a subtle 'Cancel' text if they want to stop processing */}
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-tighter">Please wait</span>
                </div>
              )}

              {/* MIC BUTTON */}
              <button 
                onClick={toggleListening}
                className={`p-3 sm:p-4 rounded-full transition-all z-40 ${
                  isListening 
                    ? 'bg-rose-500 text-white shadow-lg ring-4 ring-rose-100 scale-105' 
                    : isTranscribing 
                      ? 'opacity-0 scale-50 pointer-events-none' // Smoothly hide it
                      : 'bg-white text-emerald-600 shadow-sm hover:bg-emerald-50'
                }`}
              >
                {isListening ? <X size={24} /> : <Mic size={24} />}
              </button>
              
              <textarea 
                rows={1}
                value={input} 
                disabled={isListening || isTranscribing}
                onChange={(e) => setInput(e.target.value)} 
                onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSend())}
                placeholder={isListening ? "" : "Ask me anything..."} 
                className={`flex-1 bg-transparent border-none focus:ring-0 text-slate-800 py-3 sm:py-4 text-base sm:text-lg resize-none max-h-32 transition-opacity ${isTranscribing ? 'opacity-0' : 'opacity-100'}`}
              />

              {/* SEND BUTTON */}
              <button 
                onClick={() => handleSend()}
                disabled={!input || isListening || isTranscribing}
                className={`p-3 sm:p-4 rounded-full transition-all z-20 ${
                  !input || isListening || isTranscribing 
                    ? 'opacity-0 scale-50 pointer-events-none' 
                    : 'bg-emerald-600 text-white shadow-lg shadow-emerald-200 active:scale-90'
                }`}
              >
                <Send size={24} />
              </button>
            </div>
            <p className="text-center text-[11px] text-slate-400 mt-4 font-medium">
              Verified by Ministry of Health & Social Welfare • 2026 Updated
            </p>
          </div>
        </footer>

        {/* RIGHT PREVIEW SIDEBAR */}
        {showPreviewPanel && (
          <>
            <div
              className="fixed inset-0 bg-slate-900/25 z-40 lg:hidden"
              onClick={resetPreview}
            />
            <aside className="fixed lg:absolute right-0 top-0 bottom-0 w-full max-w-md lg:max-w-none lg:w-[27rem] bg-white border-l border-slate-200 z-50 lg:z-20 flex flex-col">
              <div className="flex items-center justify-between px-5 py-4 border-b border-slate-200 bg-slate-50/80 backdrop-blur-md">
                <div className="min-w-0">
                  <p className="text-xs font-semibold text-slate-700 truncate">
                    {previewData?.source_title || 'Reconstructed document preview'}
                  </p>
                  <p className="text-[10px] text-slate-500">
                    {previewData?.chunk_count || 0} chunks reconstructed from indexed source
                    {previewOwnerMessageIndex != null ? ` · From response #${previewOwnerMessageIndex + 1}` : ''}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {previewData?.source_url && (
                    <button
                      type="button"
                      onClick={() => window.open(previewData.source_url, '_blank')}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[10px] font-semibold bg-emerald-50 text-emerald-700 hover:bg-emerald-100 transition-colors border border-emerald-200"
                    >
                      <ExternalLink size={12} />
                      Original
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={resetPreview}
                    className="p-1.5 rounded-full text-slate-500 hover:text-slate-800 hover:bg-slate-200 transition-colors"
                    title="Close preview"
                  >
                    <X size={16} />
                  </button>
                </div>
              </div>

              <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
                {previewLoading && (
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <Loader2 size={14} className="animate-spin" />
                    Reconstructing document from chunks...
                  </div>
                )}

                {previewError && (
                  <div className="text-xs text-rose-600 bg-rose-50 border border-rose-200 rounded-lg px-3 py-2">
                    {previewError}
                  </div>
                )}

                {!previewLoading && !previewError && previewData?.chunks?.map((chunk, chunkPos) => {
                  const chunkIndex = typeof chunk.chunk_index === 'number' ? chunk.chunk_index : chunkPos;
                  const isHighlighted =
                    (
                      previewData.highlight_chunk_index != null &&
                      typeof chunk.chunk_index === 'number' &&
                      chunk.chunk_index === previewData.highlight_chunk_index
                    ) || (
                      previewData.highlight_chunk_position != null &&
                      chunkPos === previewData.highlight_chunk_position
                    );

                  const pageLabel = chunk.page_number != null
                    ? `Page ${chunk.page_number}`
                    : (chunk.page_start != null && chunk.page_end != null
                        ? `Pages ${chunk.page_start}-${chunk.page_end}`
                        : 'Page n/a');

                  return (
                    <div
                      key={`${chunkIndex}-${chunkPos}`}
                      ref={(el) => {
                        if (typeof chunk.chunk_index === 'number') {
                          previewChunkRefs.current[`idx:${chunk.chunk_index}`] = el;
                        }
                        previewChunkRefs.current[`pos:${chunkPos}`] = el;
                      }}
                      className={`rounded-xl border px-4 py-3 transition-colors ${
                        isHighlighted
                          ? 'bg-emerald-50/50 border-emerald-400 shadow-sm ring-1 ring-emerald-100'
                          : 'bg-white border-slate-200 shadow-sm hover:border-slate-300'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2 mb-1">
                        <p className="text-[10px] font-bold text-slate-600 uppercase tracking-wide">
                          Chunk {chunkIndex}
                        </p>
                        <p className="text-[10px] text-slate-500">{pageLabel}</p>
                      </div>
                      <div className="prose prose-sm max-w-none text-slate-700 leading-relaxed">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {chunk.content || ''}
                        </ReactMarkdown>
                      </div>
                    </div>
                  );
                })}
              </div>
            </aside>
          </>
        )}

        {/* DEBUG PANEL */}
        {showDebugPanel && (
          <div className="fixed right-0 top-0 bottom-0 w-full sm:w-96 bg-slate-900 text-slate-100 shadow-2xl z-50 flex flex-col border-l border-slate-700">
            <div className="p-4 border-b border-slate-700 flex items-center justify-between bg-slate-800">
              <div className="flex items-center gap-2">
                <Info size={18} className="text-purple-400" />
                <h3 className="font-bold text-sm">Debug Console</h3>
              </div>
              <button 
                onClick={() => setShowDebugPanel(false)}
                className="p-2 hover:bg-slate-700 rounded-lg transition"
              >
                <X size={18} />
              </button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-4 space-y-2 font-mono text-xs">
              {currentDebugLogs.length > 0 ? (
                currentDebugLogs.map((log, idx) => {
                  // Determine log color based on content
                  let colorClass = "text-slate-300";
                  if (log.includes("✅") || log.includes("✓")) colorClass = "text-green-400";
                  else if (log.includes("❌") || log.includes("Error")) colorClass = "text-red-400";
                  else if (log.includes("⚠️")) colorClass = "text-yellow-400";
                  else if (log.includes("🎯") || log.includes("Step")) colorClass = "text-blue-400";
                  else if (log.includes("===")) colorClass = "text-purple-400";
                  
                  return (
                    <div key={idx} className={`${colorClass} leading-relaxed whitespace-pre-wrap`}>
                      {log}
                    </div>
                  );
                })
              ) : (
                <div className="text-slate-500 text-center py-8">
                  <Info size={32} className="mx-auto mb-2 opacity-50" />
                  <p>No debug logs yet.</p>
                  <p className="text-[10px] mt-1">Send a query to see pipeline execution steps.</p>
                </div>
              )}
            </div>
            
            <div className="p-4 border-t border-slate-700 bg-slate-800">
              <button 
                onClick={() => setCurrentDebugLogs([])}
                className="w-full px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-xs font-semibold transition"
              >
                Clear Logs
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}