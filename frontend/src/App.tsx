import { useMemo, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

type AskPayload = {
  query: string
  top_k: number
  score_threshold: number | null
}

type BuildIndexResponse = {
  message: string
  indexed_chunks: number
}

type HealthResponse = {
  status: string
  indexed_chunks: number
}

type ChatRole = 'user' | 'assistant' | 'system'

type ChatMessage = {
  id: string
  role: ChatRole
  content: string
  pending?: boolean
  citations?: CitationItem[]
}

type CitationItem = {
  source: string
  chunk_id: number
  text: string
  score: number
}

type SseEnvelope = {
  event: string
  data: unknown
}

const API_PREFIX = '/api'

function App() {
  const [dataDir, setDataDir] = useState('data')
  const [forceRebuild, setForceRebuild] = useState(false)
  const [topK, setTopK] = useState(3)
  const [scoreThreshold, setScoreThreshold] = useState('')

  const [input, setInput] = useState('')
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [statusText, setStatusText] = useState('就绪')
  const [error, setError] = useState('')
  const [indexing, setIndexing] = useState(false)
  const [sending, setSending] = useState(false)
  const [overview, setOverview] = useState<OverviewResponse | null>(null)
  const [overviewLoading, setOverviewLoading] = useState(false)
  const [docKeyword, setDocKeyword] = useState('')

  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: '你好，我是你的 RAG 助手。先在左侧构建索引，然后我们开始问答。',
    },
  ])

  const chatBodyRef = useRef<HTMLDivElement | null>(null)

  const canSend = useMemo(() => input.trim().length > 0 && !sending, [input, sending])

  const addMessage = (role: ChatRole, content: string, pending = false): string => {
    const id = crypto.randomUUID()
    setMessages((prev) => [...prev, { id, role, content, pending }])
    queueMicrotask(() => {
      chatBodyRef.current?.scrollTo({ top: chatBodyRef.current.scrollHeight, behavior: 'smooth' })
    })
    return id
  }

  const updateMessage = (id: string, patch: Partial<ChatMessage>) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)))
  }

  const toDataRelativeSource = (source: string): string => {
    const marker = 'data/'
    const index = source.lastIndexOf(marker)
    if (index >= 0) {
      return source.slice(index)
    }
    return source
  }

  const parseSseEvents = (buffer: string): { events: SseEnvelope[]; rest: string } => {
    const parts = buffer.split('\n\n')
    const rest = parts.pop() ?? ''
    const events: SseEnvelope[] = []

    for (const block of parts) {
      const lines = block.split('\n')
      let eventName = ''
      let dataText = ''

      for (const line of lines) {
        if (line.startsWith('event:')) {
          eventName = line.slice(6).trim()
        } else if (line.startsWith('data:')) {
          dataText += line.slice(5).trim()
        }
      }

      if (!eventName || !dataText) continue

      try {
        events.push({ event: eventName, data: JSON.parse(dataText) })
      } catch {
        continue
      }
    }

    return { events, rest }
  }

  const parseThreshold = (): number | null => {
    if (!scoreThreshold.trim()) return null
    const value = Number(scoreThreshold)
    if (Number.isNaN(value) || value < 0 || value > 1) {
      throw new Error('score_threshold 必须是 0 到 1 之间的数字')
    }
    return value
  }

  const makeAskPayload = (query: string): AskPayload => ({
    query,
    top_k: topK,
    score_threshold: parseThreshold(),
  })

  const readError = async (resp: Response): Promise<string> => {
    try {
      const data = (await resp.json()) as { detail?: string }
      return data.detail || `请求失败 (${resp.status})`
    } catch {
      return `请求失败 (${resp.status})`
    }
  }

  const clearBanner = () => {
    setError('')
    setStatusText('处理中...')
  }

  const checkHealth = async () => {
    clearBanner()
    try {
      const resp = await fetch(`${API_PREFIX}/health`)
      if (!resp.ok) throw new Error(await readError(resp))
      const data = (await resp.json()) as HealthResponse
      setHealth(data)
      setStatusText(`后端在线，索引 chunks: ${data.indexed_chunks}`)
    } catch (err) {
      const msg = err instanceof Error ? err.message : '健康检查失败'
      setError(msg)
      setStatusText('检查失败')
    }
  }

  const buildIndex = async (event: FormEvent) => {
    event.preventDefault()
    clearBanner()
    setIndexing(true)

    try {
      const resp = await fetch(`${API_PREFIX}/index`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data_dir: dataDir, force_rebuild: forceRebuild }),
      })
      if (!resp.ok) throw new Error(await readError(resp))
      const data = (await resp.json()) as BuildIndexResponse
      setHealth({ status: 'ok', indexed_chunks: data.indexed_chunks })
      setStatusText(`${data.message}，chunks: ${data.indexed_chunks}`)
      addMessage('system', `索引已更新：${data.indexed_chunks} chunks。`) 
    } catch (err) {
      const msg = err instanceof Error ? err.message : '构建索引失败'
      setError(msg)
      setStatusText('索引失败')
    } finally {
      setIndexing(false)
    }
  }

  const sendMessage = async (event: FormEvent) => {
    event.preventDefault()
    const query = input.trim()
    if (!query || sending) return

    setInput('')
    clearBanner()
    setSending(true)

    addMessage('user', query)
    const assistantId = addMessage('assistant', '', true)

    try {
      const resp = await fetch(`${API_PREFIX}/ask/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(makeAskPayload(query)),
      })

      if (!resp.ok) throw new Error(await readError(resp))
      if (!resp.body) throw new Error('流式响应为空')

      const reader = resp.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let result = ''
      let sseBuffer = ''
      let finished = false
      let citations: CitationItem[] = []

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        sseBuffer += decoder.decode(value, { stream: true })
        const { events, rest } = parseSseEvents(sseBuffer)
        sseBuffer = rest

        for (const item of events) {
          if (item.event === 'token') {
            const payload = item.data as { text?: string }
            if (payload.text) {
              result += payload.text
              updateMessage(assistantId, { content: result, pending: true })
            }
          } else if (item.event === 'citations') {
            const payload = item.data as { items?: CitationItem[] }
            citations = Array.isArray(payload.items) ? payload.items : []
          } else if (item.event === 'error') {
            const payload = item.data as { message?: string }
            throw new Error(payload.message || '流式问答失败')
          } else if (item.event === 'done') {
            finished = true
          }
        }

        chatBodyRef.current?.scrollTo({ top: chatBodyRef.current.scrollHeight, behavior: 'smooth' })
      }

      if (!finished && !result) {
        throw new Error('流式响应未正常完成')
      }

      updateMessage(assistantId, {
        content: result || '资料中没有相关信息。',
        pending: false,
        citations,
      })
      if (citations.length > 0 && !selectedCitation) {
        setSelectedCitation(citations[0])
      }
      setStatusText('回答完成')
    } catch (err) {
      const msg = err instanceof Error ? err.message : '问答失败'
      updateMessage(assistantId, { content: `请求失败：${msg}`, pending: false })
      setError(msg)
      setStatusText('问答失败')
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="panel">
          <h2>RAG 设置</h2>
          <p className="muted">管理知识库与检索参数</p>

          <button type="button" className="ghost" onClick={checkHealth}>
            检查后端状态
          </button>
          <p className="status">{health ? `在线 · chunks ${health.indexed_chunks}` : '未检查'}</p>

          <form onSubmit={buildIndex} className="stack">
            <label>
              文档目录
              <input value={dataDir} onChange={(e) => setDataDir(e.target.value)} placeholder="data" />
            </label>

            <label className="check">
              <input
                type="checkbox"
                checked={forceRebuild}
                onChange={(e) => setForceRebuild(e.target.checked)}
              />
              强制重建索引
            </label>

            <button type="submit" disabled={indexing}>
              {indexing ? '构建中...' : '构建索引'}
            </button>
          </form>

          <div className="stack">
            <label>
              top_k
              <input
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value || 3))}
              />
            </label>

            <label>
              score_threshold
              <input
                value={scoreThreshold}
                onChange={(e) => setScoreThreshold(e.target.value)}
                placeholder="可选，如 0.3"
              />
            </label>
          </div>
        </div>
      </aside>

      <main className="chat">
        <header className="chat-header">
          <div>
            <h1>Knowledge Chat</h1>
            <p>{statusText}</p>
          </div>
          <div className="chat-header-actions">
            <button type="button" className="ghost" onClick={() => setViewerOpen((v) => !v)}>
              {viewerOpen ? '折叠文档查看' : '展开文档查看'}
            </button>
            {error && <span className="error-chip">{error}</span>}
          </div>
        </header>

        <div className={`chat-main ${viewerOpen ? 'viewer-open' : ''}`}>
          <div className="chat-body" ref={chatBodyRef}>
            {messages.map((msg) => (
              <article key={msg.id} className={`bubble ${msg.role}`}>
                <div className="role">{msg.role === 'user' ? '你' : msg.role === 'assistant' ? '助手' : '系统'}</div>
                <div className="content">{msg.content || (msg.pending ? '正在生成...' : '')}</div>
                {msg.citations && msg.citations.length > 0 && (
                  <div className="citation-links">
                    {msg.citations.map((item, idx) => (
                      <button
                        type="button"
                        key={`${item.source}-${item.chunk_id}-${idx}`}
                        className="citation-link"
                        onClick={() => {
                          setSelectedCitation(item)
                          setViewerOpen(true)
                        }}
                        title={`${toDataRelativeSource(item.source)} · chunk ${item.chunk_id}`}
                      >
                        [{idx + 1}] {toDataRelativeSource(item.source)}
                      </button>
                    ))}
                  </div>
                )}
              </article>
            ))}
          </div>

          <aside className={`doc-viewer ${viewerOpen ? 'open' : ''}`}>
            <div className="doc-viewer-header">
              <h3>文档查看</h3>
              <button type="button" className="ghost" onClick={() => setViewerOpen(false)}>
                折叠
              </button>
            </div>
            {selectedCitation ? (
              <div className="doc-viewer-body">
                <p className="doc-viewer-source">
                  {toDataRelativeSource(selectedCitation.source)} · chunk {selectedCitation.chunk_id}
                </p>
                <p className="doc-viewer-score">score: {selectedCitation.score.toFixed(3)}</p>
                <pre>{selectedCitation.text}</pre>
              </div>
            ) : (
              <p className="tiny">暂无可预览引用，先提问获取引用信息。</p>
            )}
          </aside>
        </div>

        <form className="chat-input" onSubmit={sendMessage}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="输入问题，按 Enter 发送，Shift+Enter 换行"
            rows={3}
            onKeyDown={(e) => {
              if (e.nativeEvent.isComposing) {
                return
              }
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                void sendMessage(e)
              }
            }}
          />
          <button type="submit" disabled={!canSend}>
            {sending ? '发送中...' : '发送'}
          </button>
        </form>
      </main>
    </div>
  )
}

export default App
