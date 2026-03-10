"use client";
import React, { useState, useEffect, useRef } from 'react';
import { 
  Mic, Send, Volume2, Languages, HeartPulse, Scale, 
  ShieldCheck, History, Menu, X, PlusCircle, 
  ExternalLink, FileText, Search, Loader2, Info
} from 'lucide-react';

export default function InclusiveApp() {
  const [input, setInput] = useState("");
  const [language, setLanguage] = useState("ms-MY");
  const [messages, setMessages] = useState<{ role: string; text: string; source?: string; dialect?: string; status?: string }[]>([]);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new message
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  useEffect(() => {
    const savedMessages = localStorage.getItem('chat_history');
    if (savedMessages) {
      setMessages(JSON.parse(savedMessages));
    } else {
      setMessages([{ 
        role: 'assistant', 
        text: "Selamat Datang! I am your Public Service AI. I can explain government policies in simple language or local dialects.",
        dialect: "Standard Malay"
      }]);
    }
  }, []);
const startListening = () => {
    // Check for browser support (Chrome, Edge, Safari support webkitSpeechRecognition)
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
      alert("Your browser does not support voice input. Please try Chrome or Edge.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = language; // Uses the state variable ms-MY or en-US
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onstart = () => {
      setIsListening(true);
    };

    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      setInput(transcript);
      setIsListening(false);
      // Optional: Automatically send the message after speaking
      // handleSend(transcript); 
    };

    recognition.onerror = (event: any) => {
      console.error("Speech recognition error:", event.error);
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognition.start();
  };
  const handleSend = (overrideInput?: string) => {
    const textToSend = overrideInput || input;
    if (!textToSend) return;

    const newMessages = [...messages, { role: 'user', text: textToSend }];
    setMessages(newMessages);
    setInput("");
    setIsTyping(true);
    
    // Simulate RAG + Processing
    setTimeout(() => {
      setIsTyping(false);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        text: "Summary: You can get financial assistance for medical expenses. Please visit your nearest Social Welfare Department (JKM). Bring your MyKad and latest payslip.",
        source: "https://www.moh.gov.my/policy_health_subsidy_2024.pdf",
        dialect: "Simplified Malay (Basahan)"
      }]);
    }, 1500);
  };

  return (
    <div className="flex h-screen bg-[#F8FAFC] font-sans text-slate-900 antialiased">
      
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
          <div className="flex justify-between items-center mb-8">
            <div className="flex items-center gap-2">
              <div className="bg-emerald-600 p-2 rounded-lg text-white">
                <ShieldCheck size={24} />
              </div>
              <h2 className="font-bold text-slate-800 text-xl tracking-tight">Citizen Portal</h2>
            </div>
            <button onClick={() => setIsSidebarOpen(false)} className="lg:hidden p-2 text-slate-400 hover:bg-slate-100 rounded-full transition"><X /></button>
          </div>
          
          <button 
            onClick={() => {setMessages([]); setIsSidebarOpen(false);}} 
            className="flex items-center justify-center gap-2 w-full py-3 px-4 mb-6 rounded-xl bg-slate-900 text-white font-semibold hover:bg-slate-800 transition-all shadow-sm active:scale-95"
          >
            <PlusCircle size={18}/> New Session
          </button>

          <div className="flex-1 overflow-y-auto">
            <p className="text-[11px] font-bold text-slate-400 uppercase tracking-[0.15em] mb-4">Recent Conversations</p>
            <div className="space-y-3">
               <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 text-sm text-slate-500 cursor-pointer hover:border-emerald-300 transition-colors">
                 <p className="font-semibold text-slate-700 mb-1">Health Subsidy Query</p>
                 <p className="text-xs opacity-70">2 hours ago</p>
               </div>
            </div>
          </div>
        </div>
      </aside>

      <main className="flex-1 flex flex-col min-w-0 bg-white relative">
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
          
          <div className="flex items-center gap-2 bg-slate-100 p-1 rounded-2xl border border-slate-200">
            {['ms-MY', 'en-US'].map((lang) => (
              <button 
                key={lang}
                onClick={() => setLanguage(lang)} 
                className={`px-4 py-1.5 rounded-xl text-xs font-bold transition-all ${language === lang ? 'bg-white text-emerald-700 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
              >
                {lang === 'ms-MY' ? 'Bahasa' : 'English'}
              </button>
            ))}
          </div>
        </header>

        {/* MESSAGES AREA */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 lg:p-10 space-y-8 scroll-smooth">
          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2 duration-300`}>
              <div className={`relative max-w-[90%] lg:max-w-[70%] p-5 lg:p-7 rounded-[2rem] ${
                m.role === 'user' 
                ? 'bg-emerald-600 text-white rounded-br-none shadow-lg shadow-emerald-100' 
                : 'bg-slate-50 text-slate-800 border border-slate-100 rounded-bl-none'
              }`}>
                
                {m.role === 'assistant' && m.dialect && (
                  <div className="flex items-center gap-1.5 mb-3 text-[10px] font-bold uppercase tracking-widest text-emerald-700">
                    <Languages size={14} className="opacity-70" /> {m.dialect}
                  </div>
                )}

                <p className="text-base lg:text-lg leading-relaxed font-medium">
                  {m.text}
                </p>

                {m.role === 'assistant' && (
                  <div className="mt-6 space-y-4">
                    {m.source && (
                      <div className="bg-white border border-slate-200 rounded-2xl p-4 flex items-center justify-between group hover:border-emerald-400 transition-all cursor-pointer shadow-sm">
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
                          if (isSpeaking) {
                            window.speechSynthesis.cancel();
                            setIsSpeaking(false);
                          } else {
                            window.speechSynthesis.cancel(); // Clear queue
                            const utterance = new SpeechSynthesisUtterance(m.text);
                            utterance.lang = language;
                            
                            utterance.onstart = () => setIsSpeaking(true);
                            utterance.onend = () => setIsSpeaking(false);
                            utterance.onerror = () => setIsSpeaking(false);

                            window.speechSynthesis.speak(utterance);
                          }
                        }} 
                        className={`flex items-center gap-2 px-5 py-2.5 rounded-full text-sm font-bold transition-all shadow-sm border ${
                          isSpeaking 
                            ? 'bg-rose-50 border-rose-200 text-rose-700 hover:bg-rose-100' 
                            : 'bg-white border-slate-200 text-slate-700 hover:border-emerald-500 hover:text-emerald-700'
                        }`}
                      >
                        {isSpeaking ? (
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
                      <button className="flex items-center gap-2 bg-white border border-slate-200 text-slate-700 px-5 py-2.5 rounded-full text-sm font-bold hover:border-emerald-500 transition-all shadow-sm">
                        <PlusCircle size={18} /> Simpler Version
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
        <footer className="p-4 lg:p-8 bg-white border-t border-slate-100">
          <div className="max-w-4xl mx-auto">
            {/* Quick Suggestions */}
            <div className="flex gap-2 mb-6 overflow-x-auto no-scrollbar pb-2">
              <button onClick={() => handleSend("Tell me about medical aid")} className="shrink-0 flex items-center gap-2 bg-slate-50 border border-slate-200 px-4 py-2 rounded-xl text-sm font-semibold text-slate-600 hover:bg-emerald-50 hover:border-emerald-200 transition-colors"><HeartPulse size={16}/> Medical Aid</button>
              <button onClick={() => handleSend("Welfare application")} className="shrink-0 flex items-center gap-2 bg-slate-50 border border-slate-200 px-4 py-2 rounded-xl text-sm font-semibold text-slate-600 hover:bg-blue-50 hover:border-blue-200 transition-colors"><ShieldCheck size={16}/> Welfare</button>
              <button onClick={() => handleSend("How to get legal help?")} className="shrink-0 flex items-center gap-2 bg-slate-50 border border-slate-200 px-4 py-2 rounded-xl text-sm font-semibold text-slate-600 hover:bg-amber-50 hover:border-amber-200 transition-colors"><Scale size={16}/> Legal Help</button>
            </div>

            <div className="relative flex items-end gap-3 bg-slate-50 border-2 border-slate-200 rounded-[2rem] p-2 focus-within:border-emerald-500 transition-all shadow-inner">
              <button 
                onClick={startListening}
                className={`p-4 rounded-full transition-all ${isListening ? 'bg-rose-500 text-white animate-pulse' : 'bg-white text-emerald-600 shadow-sm hover:bg-emerald-50'}`}
              >
                <Mic size={24} />
              </button>
              
              <textarea 
                rows={1}
                value={input} 
                onChange={(e) => setInput(e.target.value)} 
                onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSend())}
                placeholder="Ask me anything..." 
                className="flex-1 bg-transparent border-none focus:ring-0 text-slate-800 py-4 text-lg resize-none max-h-32"
              />

              <button 
                onClick={() => handleSend()}
                disabled={!input}
                className={`p-4 rounded-full transition-all ${!input ? 'text-slate-300' : 'bg-emerald-600 text-white shadow-lg shadow-emerald-200 active:scale-90'}`}
              >
                <Send size={24} />
              </button>
            </div>
            <p className="text-center text-[11px] text-slate-400 mt-4 font-medium">
              Verified by Ministry of Health & Social Welfare • 2026 Updated
            </p>
          </div>
        </footer>
      </main>
    </div>
  );
}