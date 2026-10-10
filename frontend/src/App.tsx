// Initial application setup + homepage prototype

type Mode = 'diabetes' | 'kidney'

const STEPS = [
  'Select Training Data',
  'Train AI',
  'Evaluation Data',
  'Evaluation Results',
  'Select Test Patient',
  'AI Detection',
  'Prediction Result',
]

// ── Header ─────────────────────────────────────────────────────
function Header({ onHome }: { onHome: () => void }) {
  return (
    <header className="bg-white/90 backdrop-blur-md border-b border-slate-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <button onClick={onHome} className="flex items-center gap-2.5 group">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-blue-700 flex items-center justify-center shadow-sm">
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
              <circle cx="9" cy="9" r="3.5" fill="white" />
              <circle cx="9" cy="3" r="1.5" fill="white" opacity="0.7" />
              <circle cx="9" cy="15" r="1.5" fill="white" opacity="0.7" />
              <circle cx="3" cy="9" r="1.5" fill="white" opacity="0.7" />
              <circle cx="15" cy="9" r="1.5" fill="white" opacity="0.7" />
              <line x1="9" y1="4.5" x2="9" y2="5.5" stroke="white" strokeWidth="1.2" opacity="0.5" />
              <line x1="9" y1="12.5" x2="9" y2="13.5" stroke="white" strokeWidth="1.2" opacity="0.5" />
              <line x1="4.5" y1="9" x2="5.5" y2="9" stroke="white" strokeWidth="1.2" opacity="0.5" />
              <line x1="12.5" y1="9" x2="13.5" y2="9" stroke="white" strokeWidth="1.2" opacity="0.5" />
            </svg>
          </div>
          <span className="font-bold text-slate-800 group-hover:text-blue-600 transition-colors" style={{ fontFamily: 'Outfit, sans-serif', fontSize: '17px' }}>Interactive AI for Healthcare</span>
        </button>
        <nav className="flex items-center gap-6">
          <button onClick={onHome} className="text-sm font-medium text-slate-600 hover:text-blue-600 transition-colors">Home</button>
          <button className="text-sm font-medium text-slate-600 hover:text-blue-600 transition-colors">About</button>
          <button className="text-sm font-medium text-slate-600 hover:text-blue-600 transition-colors">How It Works</button>
        </nav>
      </div>
    </header>
  )
}

// ── Screen 1 — Home ────────────────────────────────────────────
function Screen1({ onSelect }: { onSelect: (mode: Mode) => void }) {
  return (
    <main className="max-w-7xl mx-auto px-6 py-12">
      {/* Hero */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center mb-16 animate-fade-in">
        <div>
          <div className="inline-flex items-center gap-2 bg-blue-50 text-blue-700 px-3 py-1.5 rounded-full text-sm font-medium mb-5 border border-blue-100">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="6" stroke="#2563EB" strokeWidth="1.5" /><path d="M7 4v4l2 1.5" stroke="#2563EB" strokeWidth="1.5" strokeLinecap="round" /></svg>
            University AI Education Platform
          </div>
          <h1 style={{ fontFamily: 'Outfit, sans-serif', fontSize: 'clamp(2rem, 4vw, 3.2rem)', lineHeight: 1.15, fontWeight: 800 }} className="text-slate-900 mb-5">
            Explore How AI<br />
            <span className="text-blue-600">Learns from Healthcare</span><br />
            Data
          </h1>
          <p className="text-slate-500 text-lg leading-relaxed mb-8 max-w-lg">
            Build and test an AI model using patient data and discover how artificial intelligence can support healthcare analysis.
          </p>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-sm text-slate-500">
              <div className="w-2 h-2 rounded-full bg-blue-500"></div>
              Training data
            </div>
            <div className="flex items-center gap-2 text-sm text-slate-500">
              <div className="w-2 h-2 rounded-full bg-orange-500"></div>
              Evaluation data
            </div>
            <div className="flex items-center gap-2 text-sm text-slate-500">
              <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
              AI predictions
            </div>
          </div>
        </div>
        {/* Hero illustration */}
        <div className="flex items-center justify-center">
          <HeroIllustration />
        </div>
      </div>

      {/* Choose module */}
      <div className="text-center mb-8 animate-fade-in" style={{ animationDelay: '0.1s' }}>
        <h2 style={{ fontFamily: 'Outfit, sans-serif' }} className="text-2xl font-bold text-slate-800 mb-2">Choose Your AI Module</h2>
        <p className="text-slate-500">Select a healthcare scenario to begin your machine-learning journey.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto animate-fade-in" style={{ animationDelay: '0.2s' }}>
        {/* Diabetes card */}
        <button onClick={() => onSelect('diabetes')} className="group text-left bg-white rounded-2xl border border-slate-200 p-8 shadow-sm card-hover cursor-pointer focus:outline-none focus:ring-4 focus:ring-blue-100">
          <div className="w-16 h-16 rounded-2xl bg-blue-50 flex items-center justify-center mb-5 group-hover:bg-blue-100 transition-colors">
            <DiabetesIcon />
          </div>
          <h3 style={{ fontFamily: 'Outfit, sans-serif' }} className="text-xl font-bold text-slate-900 mb-3 group-hover:text-blue-600 transition-colors">Diabetes AI Detection</h3>
          <p className="text-slate-500 text-sm leading-relaxed mb-5">Learn how blood-test values become numerical features for a linear model that predicts diabetes.</p>
          <div className="flex items-center text-blue-600 text-sm font-semibold group-hover:gap-2 transition-all gap-1">
            Start Learning
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" className="group-hover:translate-x-1 transition-transform"><path d="M 3 8 H 13 M 9 4 L 13 8 L 9 12" stroke="#2563EB" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
          </div>
        </button>

        {/* Kidney tumour card */}
        <button onClick={() => onSelect('kidney')} className="group text-left bg-white rounded-2xl border border-slate-200 p-8 shadow-sm card-hover cursor-pointer focus:outline-none focus:ring-4 focus:ring-blue-100">
          <div className="w-16 h-16 rounded-2xl bg-red-50 flex items-center justify-center mb-5 group-hover:bg-red-100 transition-colors">
            <KidneyIcon />
          </div>
          <h3 style={{ fontFamily: 'Outfit, sans-serif' }} className="text-xl font-bold text-slate-900 mb-3 group-hover:text-red-600 transition-colors">Kidney Tumour Detection</h3>
          <p className="text-slate-500 text-sm leading-relaxed mb-5">Learn how a computer vision model processes kidney CT scans to predict whether a tumour is present.</p>
          <div className="flex items-center text-red-600 text-sm font-semibold group-hover:gap-2 transition-all gap-1">
            Start Learning
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" className="group-hover:translate-x-1 transition-transform"><path d="M 3 8 H 13 M 9 4 L 13 8 L 9 12" stroke="#DC2626" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
          </div>
        </button>
      </div>

      {/* How it works strip */}
      <div className="mt-16 bg-white rounded-2xl border border-slate-200 p-8 max-w-4xl mx-auto animate-fade-in" style={{ animationDelay: '0.3s' }}>
        <h3 style={{ fontFamily: 'Outfit, sans-serif' }} className="text-lg font-bold text-slate-800 mb-6 text-center">How It Works</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {STEPS.map((step, i) => (
            <div key={i} className="flex flex-col items-center text-center gap-2">
              <div className="w-10 h-10 rounded-xl bg-blue-50 border border-blue-100 flex items-center justify-center text-blue-600 font-bold text-sm" style={{ fontFamily: 'Outfit, sans-serif' }}>{i + 1}</div>
              <span className="text-xs text-slate-600 font-medium leading-tight">{step}</span>
            </div>
          ))}
        </div>
      </div>
    </main>
  )
}

function HeroIllustration() {
  return (
    <div className="relative w-[380px] h-[300px]">
      <svg width="380" height="300" viewBox="0 0 380 300" fill="none" className="animate-float" style={{ animationDelay: '0s' }}>
        {/* Background circle */}
        <circle cx="190" cy="150" r="130" fill="#EFF6FF" />
        <circle cx="190" cy="150" r="100" fill="#DBEAFE" opacity="0.5" />

        {/* Neural network nodes */}
        {[[80,100],[80,150],[80,200],[190,80],[190,130],[190,180],[190,230],[300,110],[300,165],[300,210]].map(([x,y],i) => (
          <circle key={i} cx={x} cy={y} r={i > 2 && i < 7 ? 14 : 10} fill={i > 2 && i < 7 ? '#2563EB' : '#93C5FD'} opacity={0.7 + (i % 3) * 0.1} className={i > 2 && i < 7 ? 'nn-node-center' : 'nn-node'} style={{ animationDelay: `${i * 0.2}s` }} />
        ))}

        {/* Connection lines */}
        {[[80,100,190,80],[80,100,190,130],[80,150,190,130],[80,150,190,180],[80,200,190,180],[80,200,190,230],
          [190,80,300,110],[190,130,300,110],[190,130,300,165],[190,180,300,165],[190,180,300,210],[190,230,300,210]].map(([x1,y1,x2,y2],i) => (
          <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#BFDBFE" strokeWidth="1.5" className="nn-edge" style={{ animationDelay: `${i * 0.15}s` }} />
        ))}

        {/* Medical cross */}
        <rect x="178" y="42" width="24" height="8" rx="2" fill="#2563EB" opacity="0.9" />
        <rect x="186" y="34" width="8" height="24" rx="2" fill="#2563EB" opacity="0.9" />

        {/* DNA helix suggestion */}
        <path d="M 330 240 Q 345 230 330 220 Q 345 210 330 200" stroke="#60A5FA" strokeWidth="2" fill="none" strokeLinecap="round" />
        <path d="M 345 240 Q 330 230 345 220 Q 330 210 345 200" stroke="#93C5FD" strokeWidth="2" fill="none" strokeLinecap="round" />

        {/* Patient avatars (small) */}
        <circle cx="46" cy="150" r="22" fill="#FEF3C7" stroke="#FDE68A" strokeWidth="1.5" />
        <circle cx="46" cy="144" r="8" fill="#F5C89A" />
        <circle cx="44" cy="143" r="2.5" fill="#1a1a2e" />
        <circle cx="49" cy="143" r="2.5" fill="#1a1a2e" />
        <path d="M 42 149 Q 46 152 50 149" stroke="#B06060" strokeWidth="1.2" fill="none" />

        {/* Output indicator */}
        <circle cx="340" cy="150" r="28" fill="#F0FDF4" stroke="#86EFAC" strokeWidth="2" />
        <path d="M 328 150 L 334 156 L 352 140" stroke="#16A34A" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />

        {/* Labels */}
        <text x="46" y="178" textAnchor="middle" fill="#94A3B8" fontSize="9" fontFamily="Inter,sans-serif">Patient</text>
        <text x="190" y="270" textAnchor="middle" fill="#2563EB" fontSize="9" fontFamily="Inter,sans-serif" fontWeight="600">Neural Network</text>
        <text x="340" y="184" textAnchor="middle" fill="#94A3B8" fontSize="9" fontFamily="Inter,sans-serif">Result</text>
      </svg>
    </div>
  )
}

function DiabetesIcon() {
  return (
    <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
      <path d="M16 4C16 4 10 8 10 14C10 17.31 12.69 20 16 20C19.31 20 22 17.31 22 14C22 8 16 4 16 4Z" fill="#BFDBFE" stroke="#2563EB" strokeWidth="1.5" strokeLinejoin="round" />
      <path d="M16 14C16 14 13 17 13 20C13 21.65 14.34 23 16 23C17.65 23 19 21.65 19 20C19 17 16 14 16 14Z" fill="#2563EB" opacity="0.6" />
      <circle cx="24" cy="10" r="4" fill="#DBEAFE" stroke="#2563EB" strokeWidth="1.2" />
      <text x="24" y="13" textAnchor="middle" fill="#2563EB" fontSize="6" fontWeight="700">G</text>
    </svg>
  )
}

function KidneyIcon() {
  return (
    <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
      <path d="M10 8 C6 8 4 12 4 16 C4 22 8 26 12 26 C14 26 16 24 16 22 C16 20 14.5 19 14 17 C13 14 14 10 16 10 C18 10 19 14 18 17 C17.5 19 16 20 16 22 C16 24 18 26 20 26 C24 26 28 22 28 16 C28 12 26 8 22 8 C20 8 18 9 16 9 C14 9 12 8 10 8Z" fill="#FECDD3" stroke="#DC2626" strokeWidth="1.5" strokeLinejoin="round" />
      <circle cx="22" cy="13" r="4" fill="#FCA5A5" stroke="#DC2626" strokeWidth="1.2" />
      <circle cx="22" cy="13" r="2" fill="#DC2626" opacity="0.6" />
    </svg>
  )
}

// ── Main App ───────────────────────────────────────────────────
export default function App() {
  const goHome = () => {
    // Homepage only in the initial prototype commit.
  }

  const handleSelectMode = (_mode: Mode) => {
    // The training workflow will be connected in a later commit.
  }

  return (
    <div className="min-h-screen" style={{ background: '#F8FAFC' }}>
      <Header onHome={goHome} />
      <Screen1 onSelect={handleSelectMode} />
    </div>
  )
}
