import { useState, useRef, useEffect, useCallback } from 'react'
import { v4 as uuidv4 } from 'uuid'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const API_URL = `${API_BASE}/chat`
const SOPS_URL = `${API_BASE}/sops`

// Stable thread_id for the full browser session
const THREAD_ID = uuidv4()

// Map severity string → badge CSS modifier
const SEV_CLASS = {
  low: 'sop-badge--low',
  moderate: 'sop-badge--moderate',
  high: 'sop-badge--high',
  critical: 'sop-badge--critical',
}

// Map SOP id prefix to severity (best-effort, shown while waiting for server)
// Real severity comes from the server response
const SOP_LABELS = {
  'SOP-001': { severity: 'high', category: 'Wind' },
  'SOP-002': { severity: 'moderate', category: 'UV Index' },
  'SOP-003': { severity: 'moderate', category: 'Heat' },
  'SOP-004': { severity: 'low', category: 'Rain' },
  'SOP-005': { severity: 'high', category: 'Travel' },
  'SOP-006': { severity: 'low', category: 'Travel' },
  'SOP-007': { severity: 'high', category: 'Vulnerable Groups' },
  'SOP-008': { severity: 'moderate', category: 'Children UV' },
  'SOP-009': { severity: 'critical', category: 'Severe Weather' },
  'SOP-010': { severity: 'low', category: 'Comfort' },
  'SOP-011': { severity: 'low', category: 'Exercise' },
}

const SUGGESTIONS = [
  { icon: '🚲', text: 'Is it safe to cycle in Mumbai today?' },
  { icon: '🌤️', text: 'Is it a good day for a picnic in London?' },
  { icon: '👧', text: 'Can my kids play outside in Delhi right now?' },
  { icon: '🚗', text: 'Is it safe to drive to the mountains today from Berlin?' },
  { icon: '🏃', text: 'Should I go for a run in Tokyo this morning?' },
  { icon: '⛅', text: 'What are the outdoor conditions like in Sydney today?' },
]

// ─── SOP Badge ───────────────────────────────────────────────────────────────

function SopBadge({ sopId, sopRegistry }) {
  if (!sopId) return null
  const meta = (sopRegistry && sopRegistry[sopId]) || SOP_LABELS[sopId] || { severity: 'low', category: 'Policy' }
  const cls = SEV_CLASS[meta.severity] || 'sop-badge--low'
  return (
    <span
      className={`sop-badge ${cls}`}
      title={`Matched policy rule: ${sopId} · Category: ${meta.category} · Severity: ${meta.severity}`}
    >
      <span className="sop-badge__dot" />
      {sopId}
    </span>
  )
}

// ─── Weather data panel ───────────────────────────────────────────────────────

function WeatherPanel({ weatherUsed }) {
  const [open, setOpen] = useState(false)
  if (!weatherUsed || Object.keys(weatherUsed).length === 0) return null

  const FIELD_UNITS = {
    temperature_2m: '°C',
    apparent_temperature: '°C',
    relative_humidity_2m: '%',
    wind_speed_10m: ' km/h',
    wind_gusts_10m: ' km/h',
    precipitation: ' mm',
    rain: ' mm',
    showers: ' mm',
    snowfall: ' cm',
    weather_code: '',
    cloud_cover: '%',
    uv_index: '',
    visibility: ' m',
    surface_pressure: ' hPa',
  }

  return (
    <div className="weather-panel">
      <button
        className="weather-panel__toggle"
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
      >
        <span className={`weather-panel__arrow ${open ? 'weather-panel__arrow--open' : ''}`}>▶</span>
        Weather data used in this reply ({Object.keys(weatherUsed).length} fields)
      </button>
      {open && (
        <div className="weather-panel__grid" role="list">
          {Object.entries(weatherUsed).map(([key, val]) => (
            <div key={key} className="weather-field" role="listitem">
              <span className="weather-field__key">{key}</span>
              <span className="weather-field__val">
                {typeof val === 'number' ? val.toFixed(1) : String(val)}
                {FIELD_UNITS[key] ?? ''}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── Individual message ───────────────────────────────────────────────────────

function Message({ msg, sopRegistry }) {
  const isUser = msg.role === 'user'
  const isSystem = msg.role === 'system'

  if (isSystem) {
    return (
      <div className="system-message" role="status">
        <div className="system-message__content">
          <span className="system-message__icon">⛅</span>
          <div className="system-message__title">Weather Advisory Bot</div>
          <p className="system-message__subtitle">{msg.text}</p>
        </div>
      </div>
    )
  }

  return (
    <div className={`message message--${isUser ? 'user' : 'bot'}`} role="listitem">
      <div className="message__avatar" aria-hidden="true">
        {isUser ? 'You' : '🤖'}
      </div>
      <div className="message__body">
        <div className="message__bubble">
          {!isUser && !msg.text ? (
            <div className="typing-indicator" aria-label="Thinking...">
              <span /><span /><span />
            </div>
          ) : (
            <p className="message__text">
              {msg.text}
              {msg.isStreaming && <span className="streaming-cursor">▌</span>}
            </p>
          )}
          {!isUser && msg.weatherUsed && (
            <WeatherPanel weatherUsed={msg.weatherUsed} />
          )}
        </div>
        {!isUser && msg.sopId && (
          <div className="message__meta">
            <SopBadge sopId={msg.sopId} sopRegistry={sopRegistry} />
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Suggestion pills ─────────────────────────────────────────────────────────

function Suggestions({ onSelect }) {
  return (
    <div className="suggestions" role="list" aria-label="Example questions">
      <span className="suggestions__label">Try asking</span>
      {SUGGESTIONS.map(s => (
        <button
          key={s.text}
          className="suggestion-pill"
          onClick={() => onSelect(s.text)}
          role="listitem"
        >
          <span aria-hidden="true">{s.icon}</span>
          {s.text}
        </button>
      ))}
    </div>
  )
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'system',
      text: 'Ask me whether it\'s safe to do an outdoor activity today. I\'ll check live weather data and match it against written safety policies — every answer is traceable to a specific rule.',
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [showSuggestions, setShowSuggestions] = useState(true)
  // Live SOP registry fetched from /sops on mount
  const [sopRegistry, setSopRegistry] = useState(null)
  const bottomRef = useRef(null)
  const textareaRef = useRef(null)

  // Fetch SOP registry from backend on mount (graceful fail)
  useEffect(() => {
    fetch(SOPS_URL)
      .then(r => r.ok ? r.json() : null)
      .then(sops => {
        if (!sops) return
        const reg = {}
        sops.forEach(s => {
          reg[s.id] = { severity: s.severity, category: s.category }
        })
        setSopRegistry(reg)
      })
      .catch(() => { /* backend not up yet, static fallback will be used */ })
  }, [])

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 120) + 'px'
  }, [input])

  const API_STREAM_URL = `${API_BASE}/chat/stream`

  const sendMessage = useCallback(async (textOverride) => {
    const text = (textOverride ?? input).trim()
    if (!text || loading) return

    // Hide suggestions after first real message
    setShowSuggestions(false)

    const userMsg = { id: uuidv4(), role: 'user', text, sopId: null }
    const botMsgId = uuidv4()

    setMessages(prev => [
      ...prev,
      userMsg,
      { id: botMsgId, role: 'bot', text: '', sopId: null, weatherUsed: null, isStreaming: true }
    ])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch(API_STREAM_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thread_id: THREAD_ID, message: text }),
      })

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed.startsWith('data: ')) continue
          const jsonStr = trimmed.slice(6)
          try {
            const event = JSON.parse(jsonStr)
            if (event.type === 'meta') {
              setMessages(prev => prev.map(m => m.id === botMsgId ? {
                ...m,
                sopId: event.sop_id || null,
                weatherUsed: event.weather_used || null,
              } : m))
            } else if (event.type === 'token') {
              setMessages(prev => prev.map(m => m.id === botMsgId ? {
                ...m,
                text: m.text + event.content,
              } : m))
            }
          } catch (e) {
            // Ignore JSON parse errors for incomplete chunks
          }
        }
      }
    } catch (err) {
      setMessages(prev => prev.map(m => m.id === botMsgId ? {
        ...m,
        text: `⚠️ Could not reach the advisory server.\n\nMake sure the backend is running:\n  cd backend && uvicorn main:app --reload --port 8000\n\n(${err.message})`,
      } : m))
    } finally {
      setMessages(prev => prev.map(m => m.id === botMsgId ? { ...m, isStreaming: false } : m))
      setLoading(false)
      textareaRef.current?.focus()
    }
  }, [input, loading])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const handleSuggestion = (text) => {
    setInput(text)
    textareaRef.current?.focus()
    // small delay so the textarea shows the text before sending
    setTimeout(() => sendMessage(text), 80)
  }

  return (
    <div className="app" role="main">
      {/* ── Header ── */}
      <header className="app__header" role="banner">
        <div className="header__icon" aria-hidden="true">⛅</div>
        <div className="header__text">
          <h1 className="app__title">Weather Advisory Bot</h1>
          <p className="app__subtitle">Live safety guidance · SOP-grounded · Open-Meteo powered</p>
        </div>
        <div className="header__status" aria-label="Bot status: online">
          <span className="status-dot" />
          Online
        </div>
      </header>

      {/* ── Thread ── */}
      <main
        id="chat-thread"
        className="app__thread"
        role="log"
        aria-live="polite"
        aria-label="Conversation"
      >
        {messages.map(msg => (
          <Message key={msg.id} msg={msg} sopRegistry={sopRegistry} />
        ))}
        {showSuggestions && !loading && (
          <Suggestions onSelect={handleSuggestion} />
        )}
        <div ref={bottomRef} />
      </main>

      {/* ── Input area ── */}
      <footer className="app__input-area" role="contentinfo">
        <div className="input-wrapper">
          <textarea
            ref={textareaRef}
            id="chat-input"
            className="app__input"
            rows={1}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about outdoor activity safety… e.g. Is it safe to cycle in Tokyo?"
            disabled={loading}
            aria-label="Message input"
            aria-describedby="input-hint"
          />
          <button
            id="send-button"
            className="app__send"
            onClick={() => sendMessage()}
            disabled={loading || !input.trim()}
            aria-label={loading ? 'Sending…' : 'Send message'}
          >
            {loading
              ? <span className="send-spinner" />
              : (
                <svg className="send-icon" viewBox="0 0 24 24" aria-hidden="true">
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              )
            }
          </button>
        </div>
        <p id="input-hint" className="input-hint">
          Press Enter to send · Shift+Enter for new line · All advice is traceable to a policy rule
        </p>
      </footer>
    </div>
  )
}
