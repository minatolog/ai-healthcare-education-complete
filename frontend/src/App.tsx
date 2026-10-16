import { useState, useEffect, useRef } from 'react'

// ── Types ──────────────────────────────────────────────────────
type Mode = 'diabetes' | 'kidney'
type ScreenId = 1 | 2 | 3 | 4 | 5 | 6 | 7

interface Patient {
  id: number
  name: string
  age: number
  sex: 'M' | 'F'
}

// ── Patient roster ─────────────────────────────────────────────
const ALL_PATIENTS: Patient[] = [
  { id: 1,  name: 'Emily Carter',     age: 42, sex: 'F' },
  { id: 2,  name: 'James Okonkwo',    age: 58, sex: 'M' },
  { id: 3,  name: 'Sarah Chen',       age: 35, sex: 'F' },
  { id: 4,  name: 'Michael Torres',   age: 67, sex: 'M' },
  { id: 5,  name: 'Priya Patel',      age: 51, sex: 'F' },
  { id: 6,  name: 'David Kim',        age: 44, sex: 'M' },
  { id: 7,  name: 'Fatima Hassan',    age: 39, sex: 'F' },
  { id: 8,  name: 'Robert Johnson',   age: 72, sex: 'M' },
  { id: 9,  name: 'Mei Lin',          age: 46, sex: 'F' },
  { id: 10, name: 'Carlos Rodriguez', age: 53, sex: 'M' },
  { id: 11, name: 'Amara Diallo',     age: 29, sex: 'F' },
  { id: 12, name: 'Thomas Hughes',    age: 61, sex: 'M' },
  { id: 13, name: 'Yuki Tanaka',      age: 38, sex: 'F' },
  { id: 14, name: 'Lakshmi Nair',     age: 55, sex: 'F' },
  { id: 15, name: 'Marcus Williams',  age: 48, sex: 'M' },
  { id: 16, name: 'Sophie Martin',    age: 33, sex: 'F' },
  { id: 17, name: 'Hassan Al-Rashid', age: 64, sex: 'M' },
  { id: 18, name: 'Emma Johansson',   age: 41, sex: 'F' },
  { id: 19, name: 'Benjamin Osei',    age: 57, sex: 'M' },
  { id: 20, name: 'Natasha Petrova',  age: 45, sex: 'F' },
  { id: 21, name: 'Diego Fernandez',  age: 50, sex: 'M' },
  { id: 22, name: 'Aisha Mohammed',   age: 36, sex: 'F' },
  { id: 23, name: "Patrick O'Brien",  age: 69, sex: 'M' },
  { id: 24, name: 'Lin Wei',          age: 43, sex: 'F' },
  { id: 25, name: 'Ingrid Andersen',  age: 52, sex: 'F' },
  { id: 26, name: 'Kwame Mensah',     age: 47, sex: 'M' },
  { id: 27, name: 'Isabella Costa',   age: 31, sex: 'F' },
  { id: 28, name: 'Viktor Novak',     age: 66, sex: 'M' },
  { id: 29, name: 'Zara Malik',       age: 40, sex: 'F' },
  { id: 30, name: 'Samuel Adeyemi',   age: 59, sex: 'M' },
]

const NEW_PATIENTS: Patient[] = [
  { id: 31, name: 'Maya Thompson', age: 34, sex: 'F' },
  { id: 32, name: 'Omar El-Sayed', age: 62, sex: 'M' },
  { id: 33, name: 'Lucia Bianchi', age: 49, sex: 'F' },
  { id: 34, name: 'Noah Williams', age: 56, sex: 'M' },
  { id: 35, name: 'Anika Sharma', age: 41, sex: 'F' },
]

// ── Avatar ─────────────────────────────────────────────────────
const SKINS = ['#FDDBB4','#F5C89A','#D4956A','#C68642','#8D5524','#6B3A2A']
const HAIRS = ['#2C1810','#4A3728','#8B6914','#D4A017','#1A1A2E','#C73E2E','#5C4033','#B5651D']

function PatientAvatar({ id, size = 64, ring }: { id: number; size?: number; ring?: string }) {
  const skin = SKINS[id % SKINS.length]
  const hair = HAIRS[(id * 3 + 1) % HAIRS.length]
  const style = id % 8

  const hairStyle = () => {
    switch (style) {
      case 0: return <><ellipse cx="32" cy="20" rx="18" ry="10" fill={hair} /><rect x="14" y="20" width="36" height="8" fill={hair} rx="2" /><rect x="12" y="26" width="6" height="18" fill={hair} rx="3" /><rect x="46" y="26" width="6" height="18" fill={hair} rx="3" /></>
      case 1: return <ellipse cx="32" cy="19" rx="16" ry="9" fill={hair} />
      case 2: return <circle cx="32" cy="18" r="15" fill={hair} />
      case 3: return <><rect x="16" y="13" width="32" height="10" fill={hair} rx="5" /></>
      case 4: return <><ellipse cx="32" cy="20" rx="18" ry="10" fill={hair} /><circle cx="32" cy="13" r="6" fill={hair} /></>
      case 5: return <ellipse cx="32" cy="20" rx="16" ry="8" fill={hair} />
      case 6: return <><ellipse cx="32" cy="19" rx="18" ry="9" fill={hair} /><rect x="14" y="22" width="36" height="6" fill={hair} /></>
      case 7: return <ellipse cx="32" cy="22" rx="16" ry="6" fill={hair} opacity="0.5" />
      default: return null
    }
  }

  const hasFacialHair = style === 5 && id % 2 === 0
  const hasGlasses = id % 5 === 0

  return (
    <svg width={size} height={size} viewBox="0 0 64 64" style={{ display: 'block' }}>
      {ring && <circle cx="32" cy="32" r="30" fill="none" stroke={ring} strokeWidth="3" />}
      {/* Neck */}
      <rect x="27" y="50" width="10" height="8" fill={skin} rx="2" />
      {/* Hair back */}
      {hairStyle()}
      {/* Face */}
      <ellipse cx="32" cy="36" rx="18" ry="19" fill={skin} />
      {/* Eyes */}
      <circle cx="25" cy="33" r="3" fill="white" />
      <circle cx="39" cy="33" r="3" fill="white" />
      <circle cx="25.5" cy="33.5" r="1.8" fill="#1a1a2e" />
      <circle cx="39.5" cy="33.5" r="1.8" fill="#1a1a2e" />
      <circle cx="26.2" cy="32.8" r="0.6" fill="white" />
      <circle cx="40.2" cy="32.8" r="0.6" fill="white" />
      {/* Eyebrows */}
      <path d="M 22 28 Q 25 26.5 28 28" stroke={hair} strokeWidth="1.5" fill="none" strokeLinecap="round" />
      <path d="M 36 28 Q 39 26.5 42 28" stroke={hair} strokeWidth="1.5" fill="none" strokeLinecap="round" />
      {/* Nose */}
      <circle cx="32" cy="38" r="1" fill={skin} style={{ filter: 'brightness(0.88)' }} />
      {/* Mouth */}
      <path d="M 26 44 Q 32 49 38 44" stroke="#B06060" strokeWidth="1.8" fill="none" strokeLinecap="round" />
      {/* Cheeks */}
      <circle cx="22" cy="40" r="4" fill="#F4A0A0" opacity="0.25" />
      <circle cx="42" cy="40" r="4" fill="#F4A0A0" opacity="0.25" />
      {/* Glasses */}
      {hasGlasses && <>
        <rect x="19" y="29.5" width="12" height="8" rx="2" fill="none" stroke="#6B7280" strokeWidth="1.4" />
        <rect x="33" y="29.5" width="12" height="8" rx="2" fill="none" stroke="#6B7280" strokeWidth="1.4" />
        <line x1="31" y1="33" x2="33" y2="33" stroke="#6B7280" strokeWidth="1.4" />
      </>}
      {/* Facial hair */}
      {hasFacialHair && <path d="M 24 46 Q 32 51 40 46" stroke={hair} strokeWidth="2.5" fill="none" strokeLinecap="round" />}
    </svg>
  )
}

// ── Progress Steps ─────────────────────────────────────────────
const STEPS = [
  'Select Training Data',
  'Train AI',
  'Evaluation Data',
  'Evaluation Results',
  'Select Test Patient',
  'AI Detection',
  'Prediction Result',
]

function ProgressSteps({ current }: { current: number }) {
  return (
    <div className="overflow-x-auto py-4 px-2">
      <div className="flex items-center justify-center gap-0 min-w-max mx-auto">
        {STEPS.map((label, i) => {
          const stepNum = i + 1
          const isActive = stepNum === current
          const isDone = stepNum < current
          return (
            <div key={i} className="flex items-center">
              <div className="flex flex-col items-center gap-1">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold transition-all duration-300 ${
                  isDone ? 'bg-blue-600 text-white' :
                  isActive ? 'bg-blue-600 text-white ring-4 ring-blue-100' :
                  'bg-slate-100 text-slate-400'
                }`}>
                  {isDone ? (
                    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                      <path d="M 2 7 L 5.5 10.5 L 12 3.5" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  ) : stepNum}
                </div>
                <span className={`text-xs font-medium whitespace-nowrap ${isActive ? 'text-blue-600' : isDone ? 'text-blue-400' : 'text-slate-400'}`} style={{ fontSize: '10px' }}>{label}</span>
              </div>
              {i < STEPS.length - 1 && (
                <div className={`h-0.5 w-5 xl:w-8 mx-1 mb-4 rounded-full transition-all duration-300 ${isDone ? 'bg-blue-400' : 'bg-slate-200'}`} />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function BackButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex items-center gap-2 border border-slate-200 bg-white text-slate-600 hover:text-blue-600 hover:border-blue-200 px-3 py-2 rounded-lg text-sm font-semibold shadow-sm transition-colors"
    >
      <svg width="15" height="15" viewBox="0 0 15 15" fill="none" aria-hidden="true">
        <path d="M12.5 7.5h-10M6.5 3.5l-4 4 4 4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      Back
    </button>
  )
}

function DiseaseStatusBadge({ patient, mode }: { patient: Patient; mode: Mode }) {
  const positive = hasPrediction(patient, mode)
  const label = mode === 'diabetes'
    ? (positive ? 'Diabetes' : 'No Diabetes')
    : (positive ? 'Kidney Tumour' : 'No Kidney Tumour')
  return (
    <span className={`mt-1 inline-flex rounded-full px-2 py-0.5 text-[10px] font-semibold ${
      positive ? 'bg-red-50 text-red-500 border border-red-100' : 'bg-blue-50 text-blue-600 border border-blue-100'
    }`}>
      {label}
    </span>
  )
}

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

// ── Screen 2 — Select Training Patients ───────────────────────
function Screen2({ trainingIds, mode, onToggle, onRandomSelect, onNext, onBack }: {
  trainingIds: Set<number>
  mode: Mode
  onToggle: (id: number) => void
  onRandomSelect: () => void
  onNext: () => void
  onBack: () => void
}) {
  const MAX = 20
  const count = trainingIds.size
  const pct = (count / MAX) * 100
  const [viewingPatient, setViewingPatient] = useState<Patient | null>(null)

  return (
    <div className="max-w-7xl mx-auto px-6 py-6 animate-fade-in">
      <ProgressSteps current={1} />
      <div className="mb-4"><BackButton onClick={onBack} /></div>

      <div className="text-center mb-6">
        <h1 style={{ fontFamily: 'Outfit, sans-serif' }} className="text-3xl font-bold text-slate-900 mb-2">Build Your AI Training Dataset</h1>
        <p className="text-slate-500 max-w-xl mx-auto mb-3">Select exactly <strong className="text-blue-600">20 patients</strong> to create the Training Dataset.</p>
        <div className="inline-flex items-start gap-2 bg-blue-50 border border-blue-100 rounded-lg px-4 py-2 text-sm text-blue-700 max-w-3xl text-left">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="6" stroke="#2563EB" strokeWidth="1.4" /><path d="M7 5v2.5l1.5 1" stroke="#2563EB" strokeWidth="1.4" strokeLinecap="round" /></svg>
          The 20 patients you select will be used to train the AI model. The remaining 10 patients will automatically become the Evaluation Dataset and will be used to evaluate how well your trained model performs on unseen data.
        </div>
        <div className="mt-4 flex flex-wrap items-center justify-center gap-3 text-sm font-semibold">
          <span className="bg-slate-100 text-slate-700 px-4 py-2 rounded-lg">30 Dataset Patients</span>
          <span className="text-slate-400">→</span>
          <span className="bg-blue-50 border border-blue-200 text-blue-700 px-4 py-2 rounded-lg">20 Training Patients</span>
          <span className="text-slate-400">+</span>
          <span className="bg-orange-50 border border-orange-200 text-orange-700 px-4 py-2 rounded-lg">10 Evaluation Patients</span>
        </div>
        <button
          type="button"
          onClick={onRandomSelect}
          className="mt-4 inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-5 py-2.5 rounded-xl text-sm font-semibold shadow-md transition-colors"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M2 4h2.2c3.8 0 3.8 8 7.6 8H14M11 9l3 3-3 3M2 12h2.2c1.2 0 2-.8 2.7-1.9M9 5.9C9.7 4.8 10.5 4 11.8 4H14M11 1l3 3-3 3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Randomly Select 20 Patients
        </button>
      </div>

      {/* Patient grid */}
      <div className="grid grid-cols-5 gap-3 mb-28">
        {ALL_PATIENTS.map(p => {
          const selected = trainingIds.has(p.id)
          const disabled = !selected && count >= MAX
          return (
            <div
              key={p.id}
              onClick={() => !disabled && onToggle(p.id)}
              role="button"
              tabIndex={disabled ? -1 : 0}
              aria-disabled={disabled}
              onKeyDown={event => {
                if (!disabled && (event.key === 'Enter' || event.key === ' ')) {
                  event.preventDefault()
                  onToggle(p.id)
                }
              }}
              className={`relative bg-white rounded-xl border-2 p-3 pb-9 flex flex-col items-center gap-2 transition-all duration-200 focus:outline-none ${
                selected ? 'patient-card-selected-train' :
                disabled ? 'opacity-40 cursor-not-allowed border-slate-200' :
                'border-slate-200 hover:border-blue-200 hover:shadow-md hover:-translate-y-0.5 cursor-pointer'
              }`}
            >
              {selected && (
                <div className="absolute top-1.5 right-1.5 w-5 h-5 bg-blue-600 rounded-full flex items-center justify-center checkmark-anim z-10">
                  <svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M 1.5 5 L 4 7.5 L 8.5 2.5" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
                </div>
              )}
              <PatientAvatar id={p.id} size={52} />
              <div className="text-center">
                <div className="text-xs font-semibold text-slate-800 leading-tight">{p.name}</div>
                <div className="text-xs text-slate-400 mt-0.5" style={{ fontFamily: 'DM Mono, monospace' }}>Age {p.age}</div>
                <DiseaseStatusBadge patient={p} mode={mode} />
              </div>
              <button
                type="button"
                title="View patient data"
                aria-label={`View patient data for ${p.name}`}
                onClick={event => {
                  event.stopPropagation()
                  setViewingPatient(p)
                }}
                className="absolute bottom-2 right-2 w-7 h-7 rounded-lg bg-slate-50 border border-slate-200 text-slate-400 hover:text-blue-600 hover:border-blue-200 hover:bg-blue-50 flex items-center justify-center transition-colors"
              >
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
                  <path d="M2.5 1.5h6l3 3v8h-9z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
                  <path d="M8.5 1.5v3h3M4.5 7h5M4.5 9.5h4" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
                </svg>
              </button>
            </div>
          )
        })}
      </div>

      {viewingPatient && (
        <PatientDataModal
          patient={viewingPatient}
          mode={mode}
          onClose={() => setViewingPatient(null)}
        />
      )}

      {/* Sticky bottom panel */}
      <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-slate-200 shadow-lg z-40">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center gap-6">
          <div className="flex-1">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-sm font-semibold text-slate-700">Training Patients Selected</span>
              <span className={`text-lg font-bold ${count === MAX ? 'text-blue-600' : 'text-slate-700'}`} style={{ fontFamily: 'Outfit, sans-serif' }}>
                {count} <span className="text-slate-300">/</span> {MAX}
              </span>
            </div>
            <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{ width: `${pct}%`, background: count === MAX ? 'linear-gradient(90deg, #2563EB, #60A5FA)' : 'linear-gradient(90deg, #93C5FD, #3B82F6)' }}
              />
            </div>
          </div>
          <button
            onClick={onNext}
            disabled={count !== MAX}
            className={`flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-sm transition-all duration-200 ${
              count === MAX
                ? 'bg-blue-600 text-white hover:bg-blue-700 shadow-md hover:shadow-lg active:scale-95'
                : 'bg-slate-100 text-slate-400 cursor-not-allowed'
            }`}
          >
            Next
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M 3 8 H 13 M 9 4 L 13 8 L 9 12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Screen 3 — Model training console ─────────────────────────
function makeTrainingLogs(mode: Mode) {
  const startup = mode === 'diabetes'
    ? [
        'Loading blood test dataset…',
        '20 patient records loaded.',
        'Extracting numerical features…',
        'Features: Glucose, HbA1c, Insulin, Cholesterol, Triglycerides, HDL, LDL',
        'Normalising numerical features…',
        'Preparing binary labels…',
        'Initialising linear binary classifier…',
        'Starting model training…',
      ]
    : [
        'Loading kidney CT dataset…',
        '20 patient CT datasets loaded.',
        'Reading CT images…',
        'Resizing CT images…',
        'Normalising image intensity…',
        'Preparing tumour / non-tumour labels…',
        'Initialising computer vision model…',
        'Starting model training…',
      ]
  const epochs = Array.from({ length: 20 }, (_, index) => {
    const epoch = index + 1
    const startLoss = mode === 'diabetes' ? 0.6842 : 0.7214
    const endLoss = mode === 'diabetes' ? 0.2147 : 0.1842
    const loss = startLoss - ((startLoss - endLoss) * index) / 19
    const accuracy = 0.65 + (0.95 - 0.65) * index / 19
    return `Epoch ${epoch}/20  loss: ${loss.toFixed(4)} - accuracy: ${accuracy.toFixed(4)}`
  })
  return [
    ...startup,
    ...epochs,
    'Finalising trained model parameters…',
    '✓ Training complete',
    mode === 'diabetes' ? '✓ Model saved successfully' : '✓ Computer vision model saved successfully',
  ]
}

const TRAINING_CONCEPTS = [
  {
    term: 'Epoch',
    title: 'Epoch – One Learning Round',
    explanation: 'An epoch means the AI has gone through all 20 training patients once. If we train for 20 epochs, the AI studies the same training dataset 20 times to improve its learning.',
    example: 'Epoch 5/20 means the AI has completed 5 out of 20 learning rounds.',
  },
  {
    term: 'Loss',
    title: 'Loss – How Wrong Is the AI?',
    explanation: "Loss measures how far the AI's predictions are from the correct answers. A smaller loss usually means the AI is learning to make better predictions.",
    example: 'Loss: 0.80 → 0.30 means the AI is making fewer prediction errors according to its loss function.',
  },
  {
    term: 'Accuracy',
    title: 'Accuracy – How Many Answers Are Correct?',
    explanation: 'Accuracy tells us the percentage of predictions the AI gets right.',
    example: 'If the AI correctly predicts 8 out of 10 patients, its accuracy is 80%.',
  },
  {
    term: 'Training Data',
    title: 'Training Data – Learning from Examples',
    explanation: 'The AI learns from 20 selected patients. Each patient has medical information and a known diagnosis, helping the AI discover patterns associated with the disease.',
  },
  {
    term: 'Evaluation Data',
    title: 'Evaluation Data – Testing What AI Has Learned',
    explanation: 'The remaining 10 patients are kept separate from training. After training, we use them to check how well the AI performs on patients it has not learned from.',
  },
  {
    term: 'AI Model',
    title: 'AI Model – The Pattern Finder',
    explanation: 'An AI model is a computer program that learns patterns from examples. After training, it uses these patterns to make predictions about new patients.',
  },
]

function TrainingEducation() {
  return (
    <section className="mb-5">
      <div className="text-center mb-5">
        <h2 className="text-2xl font-bold text-slate-900 mb-1">Understanding How AI Learns</h2>
        <p className="text-slate-500">Let's explore what happens when we train an AI model!</p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {TRAINING_CONCEPTS.map((concept, index) => (
          <article key={concept.term} className="bg-white rounded-xl border border-blue-100 shadow-sm p-5">
            <div className="flex items-start gap-3">
              <div className="w-9 h-9 rounded-lg bg-blue-50 border border-blue-100 text-blue-600 flex items-center justify-center flex-shrink-0">
                <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
                  <circle cx="9" cy="9" r="7" stroke="currentColor" strokeWidth="1.5" />
                  <path d={index % 2 === 0 ? 'M6 9h6M9 6v6' : 'M5.5 11.5L8 9l2 1.5 3-4'} stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
              <div>
                <span className="inline-flex bg-blue-50 text-blue-700 rounded-full px-2 py-0.5 text-xs font-bold mb-2">{concept.term}</span>
                <h3 className="text-base font-bold text-slate-900 mb-2">{concept.title}</h3>
                <p className="text-sm text-slate-600 leading-relaxed">{concept.explanation}</p>
                {concept.example && <p className="mt-3 bg-slate-50 border border-slate-100 rounded-lg px-3 py-2 text-xs text-slate-600"><strong>Example:</strong> {concept.example}</p>}
              </div>
            </div>
          </article>
        ))}
      </div>
    </section>
  )
}

function Screen3({ mode, completed, initialVisibleCount, onProgress, onTrainingComplete, onContinue, onBack }: {
  mode: Mode
  completed: boolean
  initialVisibleCount: number
  onProgress: (count: number) => void
  onTrainingComplete: () => void
  onContinue: () => void
  onBack: () => void
}) {
  const logs = makeTrainingLogs(mode)
  const [visibleCount, setVisibleCount] = useState(completed ? logs.length : initialVisibleCount)
  const consoleRef = useRef<HTMLDivElement>(null)
  const done = visibleCount === logs.length

  useEffect(() => {
    if (completed) {
      setVisibleCount(logs.length)
      return
    }
    const timer = window.setInterval(() => {
      setVisibleCount(count => {
        if (count >= logs.length) {
          window.clearInterval(timer)
          return count
        }
        const nextCount = count + 1
        onProgress(nextCount)
        return nextCount
      })
    }, 180)
    return () => window.clearInterval(timer)
  }, [mode, logs.length, completed])

  useEffect(() => {
    if (consoleRef.current) consoleRef.current.scrollTop = consoleRef.current.scrollHeight
  }, [visibleCount])

  useEffect(() => {
    if (done && !completed) onTrainingComplete()
  }, [done, completed, onTrainingComplete])

  return (
    <div className="max-w-5xl mx-auto px-6 py-6 animate-fade-in">
      <ProgressSteps current={2} />
      <div className="mb-4"><BackButton onClick={onBack} /></div>
      <div className="text-center mb-8">
        <h1 style={{ fontFamily: 'Outfit, sans-serif' }} className="text-3xl font-bold text-slate-900 mb-2">AI Model Training Console</h1>
        <p className="text-slate-500">Watch each step as the model learns from your 20 selected patients.</p>
      </div>

      <div className="bg-blue-50 border border-blue-100 text-blue-800 rounded-xl px-5 py-4 mb-5 text-sm">
        <strong>The AI model is learning only from the 20 Training Dataset patients.</strong> The 10 Evaluation Dataset patients remain unseen during training.
      </div>

      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden mb-5">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-px bg-slate-200 border-b border-slate-200">
          {[
            ['Model', mode === 'diabetes' ? 'Linear Binary Classifier' : 'Computer Vision Classifier'],
            ['Input', mode === 'diabetes' ? 'Blood Test Features' : 'Kidney CT Images'],
            ['Training Dataset', '20 patients'],
            ['Evaluation Dataset', '10 reserved · unseen'],
            ['Task', mode === 'diabetes' ? 'Diabetes Classification' : 'Tumour Image Classification'],
          ].map(([label, value]) => (
            <div key={label} className="bg-white p-4">
              <div className="text-xs text-slate-400 mb-1">{label}</div>
              <div className="text-sm font-semibold text-slate-800">{value}</div>
            </div>
          ))}
        </div>
        <div className="bg-slate-950 px-5 py-4 flex items-center justify-between border-b border-slate-800">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-red-400" />
            <span className="w-3 h-3 rounded-full bg-amber-400" />
            <span className="w-3 h-3 rounded-full bg-emerald-400" />
          </div>
          <span className="font-mono text-xs text-slate-500">training_session.log</span>
        </div>
        <div ref={consoleRef} className="h-96 overflow-y-auto bg-slate-950 p-5 font-mono text-sm leading-7" aria-live="polite">
          {logs.slice(0, visibleCount).map((line, index) => (
            <div
              key={index}
              className={
                line.startsWith('✓') ? 'text-emerald-400' :
                line.startsWith('Epoch') ? 'text-sky-300' :
                line.startsWith('Features:') ? 'text-amber-300' :
                'text-slate-300'
              }
            >
              <span className="text-slate-600 select-none mr-3">$</span>{line}
            </div>
          ))}
          {!done && <span className="terminal-cursor inline-block w-2 h-4 bg-emerald-400 align-middle" />}
        </div>
      </div>

      <TrainingEducation />

      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <span className={`w-3 h-3 rounded-full ${done ? 'bg-emerald-500' : 'bg-blue-500 animate-pulse'}`} />
          <div>
            <div className={`text-sm font-semibold ${done ? 'text-emerald-700' : 'text-blue-700'}`}>
              {done ? 'AI Model Training Complete' : 'Training in progress…'}
            </div>
            <div className="text-xs text-slate-400">
              {done ? 'The trained model is ready for the reserved Evaluation Dataset.' : 'Logs are updating automatically.'}
            </div>
          </div>
        </div>
        <button
          onClick={onContinue}
          disabled={!done}
          className={`inline-flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-sm transition-all ${
            done ? 'bg-blue-600 text-white hover:bg-blue-700 shadow-md' : 'bg-slate-100 text-slate-400 cursor-not-allowed'
          }`}
        >
          Continue to Evaluation
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M 3 8 H 13 M 9 4 L 13 8 L 9 12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
        </button>
      </div>
    </div>
  )
}

// ── Screen 4 — Evaluation Data ───────────────────────────────
function ViewDataButton({ patient, onView }: { patient: Patient; onView: (patient: Patient) => void }) {
  return (
    <button
      type="button"
      title="View patient data"
      aria-label={`View patient data for ${patient.name}`}
      onClick={event => { event.stopPropagation(); onView(patient) }}
      className="absolute bottom-2 right-2 w-7 h-7 rounded-lg bg-slate-50 border border-slate-200 text-slate-400 hover:text-blue-600 hover:border-blue-200 hover:bg-blue-50 flex items-center justify-center transition-colors"
    >
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
        <path d="M2.5 1.5h6l3 3v8h-9z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
        <path d="M8.5 1.5v3h3M4.5 7h5M4.5 9.5h4" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
      </svg>
    </button>
  )
}

function EvaluationDataScreen({ patients, mode, completed, onComplete, onBack }: {
  patients: Patient[]
  mode: Mode
  completed: boolean
  onComplete: () => void
  onBack: () => void
}) {
  const [viewingPatient, setViewingPatient] = useState<Patient | null>(null)
  const [evaluating, setEvaluating] = useState(false)
  const [currentPatient, setCurrentPatient] = useState(-1)
  const evaluationMessages = mode === 'diabetes'
    ? ['Reading blood test values…', 'Normalising numerical features…', 'Running linear classifier…', 'Generating prediction…']
    : ['Loading kidney CT scan…', 'Preprocessing CT image…', 'Extracting visual features…', 'Running computer vision model…', 'Generating prediction…']

  useEffect(() => {
    if (!evaluating) return
    const timer = window.setInterval(() => {
      setCurrentPatient(current => {
        if (current >= patients.length - 1) {
          window.clearInterval(timer)
          return patients.length
        }
        return current + 1
      })
    }, 650)
    return () => window.clearInterval(timer)
  }, [evaluating, patients.length])

  useEffect(() => {
    if (currentPatient !== patients.length) return
    const timer = window.setTimeout(onComplete, 650)
    return () => window.clearTimeout(timer)
  }, [currentPatient, patients.length, onComplete])

  return (
    <div className="max-w-6xl mx-auto px-6 py-6 animate-fade-in">
      <ProgressSteps current={3} />
      <div className="mb-4"><BackButton onClick={onBack} /></div>
      <div className="text-center mb-6">
        <h1 className="text-3xl font-bold text-slate-900 mb-2">Evaluate Your AI Model</h1>
        <p className="text-slate-500">The remaining 10 patients form the Evaluation Dataset.</p>
      </div>
      <div className="bg-orange-50 border border-orange-200 rounded-xl px-5 py-4 mb-6 text-sm text-orange-900 leading-relaxed">
        <strong>These patients were not used during model training.</strong> The trained model will make predictions for all 10 patients, then compare them with their known correct labels to measure performance on unseen data.
      </div>

      {!evaluating ? (
        <>
          <div className="flex items-center gap-3 mb-5">
            <div className="flex-1 h-px bg-orange-200" />
            <div className="bg-orange-50 border border-orange-200 text-orange-700 rounded-full px-4 py-1.5 text-sm font-semibold">Evaluation Dataset — 10 Unseen Patients</div>
            <div className="flex-1 h-px bg-orange-200" />
          </div>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
            {patients.map(patient => (
              <div key={patient.id} className="relative bg-white rounded-xl border-2 border-orange-100 p-3 pb-9 flex flex-col items-center gap-2 shadow-sm">
                <span className="absolute top-2 left-2 bg-orange-50 border border-orange-100 text-orange-700 rounded-full px-2 py-0.5 text-[10px] font-semibold">Evaluation Data</span>
                <PatientAvatar id={patient.id} size={52} />
                <div className="text-center">
                  <div className="text-xs font-semibold text-slate-800 leading-tight">{patient.name}</div>
                  <div className="text-xs text-slate-400 mt-0.5 font-mono">Age {patient.age}</div>
                  <DiseaseStatusBadge patient={patient} mode={mode} />
                </div>
                <ViewDataButton patient={patient} onView={setViewingPatient} />
              </div>
            ))}
          </div>
          <div className="text-center">
            <button onClick={() => completed ? onComplete() : (setEvaluating(true), setCurrentPatient(0))} className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 rounded-xl font-bold shadow-lg transition-all active:scale-95">
              {completed ? 'View Evaluation Results' : 'Start Model Evaluation'}
              <span aria-hidden="true">→</span>
            </button>
          </div>
        </>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_1.35fr] gap-6 items-start">
          <div className="bg-gradient-to-br from-blue-50 to-white rounded-2xl border border-blue-100 p-7 text-center lg:sticky lg:top-6">
            <div className="relative w-20 h-20 mx-auto mb-4 flex items-center justify-center">
              <div className="absolute inset-0 rounded-full border-4 border-blue-100" />
              <div className="absolute inset-0 rounded-full border-4 border-blue-600 border-t-transparent animate-spin" />
              <AIReadyIcon />
            </div>
            <h2 className="text-xl font-bold text-slate-900 mb-2">Evaluating model on unseen data…</h2>
            <div className="text-3xl font-bold text-blue-600 mb-3">Patient {Math.min(currentPatient + 1, 10)} of 10</div>
            <p className="font-mono text-xs text-slate-500">{currentPatient >= patients.length ? 'Prediction complete ✓' : evaluationMessages[currentPatient % evaluationMessages.length]}</p>
          </div>
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden" aria-live="polite">
            {patients.map((patient, index) => {
              const status = index < currentPatient ? 'evaluated' : index === currentPatient ? 'analysing' : 'waiting'
              return (
                <div key={patient.id} className={`flex items-center gap-3 px-4 py-3 border-b last:border-b-0 border-slate-100 ${status === 'analysing' ? 'bg-blue-50' : ''}`}>
                  <PatientAvatar id={patient.id} size={38} />
                  <span className="text-sm font-medium text-slate-700 flex-1">{patient.name}</span>
                  {status === 'evaluated' && <span className="text-xs font-semibold text-emerald-700">✓ Evaluated</span>}
                  {status === 'analysing' && <span className="text-xs font-semibold text-blue-700 flex items-center gap-2"><span className="w-3 h-3 rounded-full border-2 border-blue-600 border-t-transparent animate-spin" />Analysing…</span>}
                  {status === 'waiting' && <span className="text-xs font-medium text-slate-400">Waiting</span>}
                </div>
              )
            })}
          </div>
        </div>
      )}
      {viewingPatient && <PatientDataModal patient={viewingPatient} mode={mode} onClose={() => setViewingPatient(null)} />}
    </div>
  )
}

// ── Screen 5 — Evaluation Results ─────────────────────────────
function getEvaluationRows(patients: Patient[], mode: Mode) {
  return patients.map((patient, index) => {
    const actualPositive = hasPrediction(patient, mode)
    const correct = index !== 1 && index !== 7
    const predictedPositive = correct ? actualPositive : !actualPositive
    const positiveLabel = mode === 'diabetes' ? 'Diabetes' : 'Kidney Tumour'
    const negativeLabel = mode === 'diabetes' ? 'No Diabetes' : 'No Kidney Tumour'
    return {
      patient,
      correct,
      actual: actualPositive ? positiveLabel : negativeLabel,
      prediction: predictedPositive ? positiveLabel : negativeLabel,
    }
  })
}

function EvaluationResultsScreen({ patients, mode, onNext, onBack }: { patients: Patient[]; mode: Mode; onNext: () => void; onBack: () => void }) {
  const rows = getEvaluationRows(patients, mode)
  return (
    <div className="max-w-6xl mx-auto px-6 py-6 animate-fade-in">
      <ProgressSteps current={4} />
      <div className="mb-4"><BackButton onClick={onBack} /></div>
      <div className="text-center mb-7">
        <h1 className="text-3xl font-bold text-slate-900 mb-2">Model Evaluation Results</h1>
        <p className="text-slate-500 max-w-2xl mx-auto">Your trained AI model has been evaluated using 10 unseen patients that were not included in the Training Dataset.</p>
        <span className="inline-block mt-3 bg-slate-100 text-slate-500 rounded-full px-3 py-1 text-xs font-medium">Demonstration evaluation results</span>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-7">
        {[
          ['Evaluation Patients', '10', 'text-slate-800'],
          ['Correct Predictions', '8 / 10', 'text-emerald-600'],
          ['Incorrect Predictions', '2 / 10', 'text-red-600'],
          ['Accuracy', '80%', 'text-blue-600'],
        ].map(([label, value, color]) => (
          <div key={label} className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 text-center">
            <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">{label}</div>
            <div className={`text-3xl font-bold ${color}`}>{value}</div>
          </div>
        ))}
      </div>

      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden mb-7">
        <div className="px-5 py-4 bg-slate-50 border-b border-slate-200">
          <h2 className="text-lg font-bold text-slate-900">Actual Labels vs AI Predictions</h2>
        </div>
        <div className="divide-y divide-slate-100">
          {rows.map(({ patient, actual, prediction, correct }) => (
            <div key={patient.id} className="grid grid-cols-[minmax(10rem,1.4fr)_1fr_1fr_auto] gap-4 items-center px-5 py-4">
              <div className="flex items-center gap-3 min-w-0">
                <PatientAvatar id={patient.id} size={42} />
                <span className="font-semibold text-sm text-slate-800 truncate">{patient.name}</span>
              </div>
              <div><div className="text-xs text-slate-400">Actual</div><div className="text-sm font-medium text-slate-700">{actual}</div></div>
              <div><div className="text-xs text-slate-400">AI Prediction</div><div className="text-sm font-medium text-slate-700">{prediction}</div></div>
              <span className={`min-w-24 text-center rounded-full px-3 py-1.5 text-xs font-bold ${correct ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-red-50 text-red-700 border border-red-200'}`}>{correct ? '✓ Correct' : '✕ Incorrect'}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-100 rounded-2xl p-6 mb-7">
        <h2 className="text-lg font-bold text-blue-900 mb-2">Why do we evaluate the model?</h2>
        <p className="text-sm text-blue-800 leading-relaxed mb-5">Evaluation Data helps us measure how well the trained model performs on patients it has never seen before. Comparing the AI predictions with the known correct labels gives us an estimate of the model’s performance.</p>
        <div className="flex flex-wrap items-center justify-center gap-2 text-xs font-semibold text-slate-700">
          <span className="bg-white border border-blue-200 rounded-lg px-3 py-2">10 Unseen Patients</span><span>→</span>
          <span className="bg-white border border-blue-200 rounded-lg px-3 py-2">Trained AI Model</span><span>→</span>
          <span className="bg-white border border-blue-200 rounded-lg px-3 py-2">AI Predictions</span><span>+</span>
          <span className="bg-white border border-blue-200 rounded-lg px-3 py-2">Actual Labels</span><span>→</span>
          <span className="bg-blue-600 text-white rounded-lg px-3 py-2">Compare → 80% Accuracy</span>
        </div>
        <div className="text-center mt-4 font-mono text-sm text-blue-900">8 correct predictions ÷ 10 evaluation patients = 80% accuracy</div>
      </div>
      <div className="text-center"><button onClick={onNext} className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-9 py-3.5 rounded-xl font-bold shadow-md transition-colors">Next <span aria-hidden="true">→</span></button></div>
    </div>
  )
}

// ── Screen 6 — Select New Test Patient ────────────────────────
function SelectTestPatientScreen({ mode, testId, onSelect, onNext, onBack }: { mode: Mode; testId: number | null; onSelect: (id: number) => void; onNext: () => void; onBack: () => void }) {
  const [viewingPatient, setViewingPatient] = useState<Patient | null>(null)
  return (
    <div className="max-w-7xl mx-auto px-6 py-6 animate-fade-in">
      <ProgressSteps current={5} />
      <div className="mb-4"><BackButton onClick={onBack} /></div>
      <div className="text-center mb-6">
        <h1 className="text-3xl font-bold text-slate-900 mb-2">Choose a New Patient for AI Prediction</h1>
        <p className="text-slate-500 max-w-2xl mx-auto mb-3">Now that the model has been trained and evaluated, choose one new patient to see how the AI makes a prediction for an individual case.</p>
        <div className="inline-flex items-center gap-2 bg-red-50 border border-red-100 rounded-lg px-4 py-2 text-sm text-red-700">These are new patient cases. They were not used in either the Training Dataset or the Evaluation Dataset.</div>
      </div>
      <div className="flex items-center gap-3 mb-5 max-w-3xl mx-auto"><div className="flex-1 h-px bg-red-100" /><div className="bg-red-50 border border-red-100 rounded-full px-4 py-1.5 text-xs font-semibold text-red-700">New Patient Cases — 5 Patients</div><div className="flex-1 h-px bg-red-100" /></div>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8 max-w-4xl mx-auto">
        {NEW_PATIENTS.map(patient => {
          const selected = testId === patient.id
          return (
            <div key={patient.id} role="button" tabIndex={0} onClick={() => onSelect(patient.id)} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(patient.id) } }} className={`relative bg-white rounded-xl border-2 p-3 pb-9 flex flex-col items-center gap-2 transition-all cursor-pointer focus:outline-none ${selected ? 'patient-card-selected-test' : 'border-slate-200 hover:border-red-200 hover:shadow-md hover:-translate-y-0.5'}`}>
              {selected && <div className="absolute top-1.5 right-1.5 w-5 h-5 bg-red-600 rounded-full flex items-center justify-center checkmark-anim"><span className="text-white text-xs">✓</span></div>}
              <PatientAvatar id={patient.id} size={52} />
              <div className="text-center"><div className="text-xs font-semibold text-slate-800">{patient.name}</div><div className="text-xs text-slate-400 mt-0.5 font-mono">Age {patient.age}</div></div>
              <ViewDataButton patient={patient} onView={setViewingPatient} />
            </div>
          )
        })}
      </div>
      <div className="text-center">
        <button onClick={onNext} disabled={!testId} className={`inline-flex items-center gap-2 px-8 py-3.5 rounded-xl font-semibold text-sm ${testId ? 'bg-red-600 text-white hover:bg-red-700 shadow-md' : 'bg-slate-100 text-slate-400 cursor-not-allowed'}`}>
          Next <span aria-hidden="true">→</span>
        </button>
      </div>
      {viewingPatient && <PatientDataModal patient={viewingPatient} mode={mode} onClose={() => setViewingPatient(null)} />}
    </div>
  )
}

// ── Patient data generation ────────────────────────────────────
function getBloodTests(patient: Patient) {
  const seed = patient.id
  return {
    'Fasting Blood Glucose': `${85 + (seed * 11) % 90} mg/dL`,
    HbA1c: `${(5.1 + ((seed * 7) % 31) / 10).toFixed(1)}%`,
    Insulin: `${10 + (seed * 5) % 25} μU/mL`,
    Cholesterol: `${165 + (seed * 9) % 70} mg/dL`,
    Triglycerides: `${90 + (seed * 13) % 100} mg/dL`,
    HDL: `${38 + (seed * 7) % 27} mg/dL`,
    LDL: `${80 + (seed * 11) % 75} mg/dL`,
  }
}

function hasPrediction(p: Patient, mode: Mode) {
  return mode === 'diabetes' ? p.id % 3 !== 1 : p.id % 3 !== 0
}

function getConfidence(p: Patient) {
  return 68 + (p.id * 7) % 25
}

function PatientProfile({ patient, test = false }: { patient: Patient; test?: boolean }) {
  return (
    <div className={`flex items-center gap-4 p-4 rounded-xl border ${test ? 'bg-red-50 border-red-100' : 'bg-blue-50 border-blue-100'}`}>
      <PatientAvatar id={patient.id} size={64} ring={test ? '#DC2626' : '#2563EB'} />
      <div>
        <h3 style={{ fontFamily: 'Outfit, sans-serif' }} className="font-bold text-slate-900 text-lg">{patient.name}</h3>
        <div className="flex flex-wrap items-center gap-2 mt-1 text-sm text-slate-500">
          <span className="font-mono">Age {patient.age}</span>
          <span>·</span>
          <span>{patient.sex === 'F' ? 'Female' : 'Male'}</span>
          {test && <span className="bg-red-100 text-red-700 px-2 py-0.5 rounded-full text-xs font-medium">Test Patient</span>}
        </div>
      </div>
    </div>
  )
}

function ModeData({ patient, mode, analysing = false }: { patient: Patient; mode: Mode; analysing?: boolean }) {
  if (mode === 'kidney') {
    return (
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="px-4 py-3 bg-slate-50 border-b border-slate-200">
          <span className="text-xs font-semibold text-slate-700 uppercase tracking-wider">Kidney CT Scan</span>
        </div>
        <div className="p-4">
          <KidneyCTScan patientId={patient.id} analysing={analysing} />
          <p className="text-xs text-slate-500 mt-3">Input image used by the computer vision model</p>
        </div>
      </div>
    )
  }
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
      <div className="px-4 py-3 bg-slate-50 border-b border-slate-200">
        <span className="text-xs font-semibold text-slate-700 uppercase tracking-wider">Blood Test Data</span>
      </div>
      <div className="divide-y divide-slate-100">
        {Object.entries(getBloodTests(patient)).map(([label, value]) => (
          <div key={label} className="flex items-center justify-between gap-4 px-4 py-3">
            <span className="text-sm text-slate-500">{label}</span>
            <span className="text-sm font-mono font-medium text-slate-800 text-right">{value}</span>
          </div>
        ))}
      </div>
      <p className="px-4 py-3 bg-blue-50 text-xs text-blue-700 border-t border-blue-100">Input data used by the linear classification model</p>
    </div>
  )
}

function PatientDataModal({ patient, mode, onClose }: { patient: Patient; mode: Mode; onClose: () => void }) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => event.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/45 p-4 flex items-center justify-center" onMouseDown={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label={`Patient data for ${patient.name}`}
        className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto bg-white rounded-2xl shadow-2xl p-6 animate-scale-in"
        onMouseDown={event => event.stopPropagation()}
      >
        <button type="button" onClick={onClose} aria-label="Close patient data" className="absolute top-4 right-4 w-9 h-9 rounded-lg bg-slate-100 text-slate-500 hover:bg-slate-200 flex items-center justify-center">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M3 3l10 10M13 3L3 13" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /></svg>
        </button>
        <div className="pr-12 mb-5"><PatientProfile patient={patient} /></div>
        <ModeData patient={patient} mode={mode} />
      </div>
    </div>
  )
}

// Shared, stable left column for detection and prediction states.
function PatientDataPanel({ patient, mode, analysing = false }: { patient: Patient; mode: Mode; analysing?: boolean }) {
  return (
    <div className="space-y-5">
      <PatientProfile patient={patient} test />
      <ModeData patient={patient} mode={mode} analysing={analysing} />
    </div>
  )
}

function KidneyCTScan({ patientId, analysing = false }: { patientId: number; analysing?: boolean }) {
  const hasTumor = patientId % 3 !== 0
  const tumorX = 120 + (patientId * 17) % 40
  const tumorY = 80 + (patientId * 11) % 30
  const tumorR = 8 + (patientId * 5) % 10

  return (
    <div className="relative bg-slate-900 rounded-xl overflow-hidden" style={{ aspectRatio: '4/3' }}>
      <svg width="100%" viewBox="0 0 280 210" className="w-full" style={{ background: '#0F172A' }}>
        {/* CT scan background */}
        <ellipse cx="140" cy="105" rx="120" ry="90" fill="#1E293B" />
        {/* Body tissue */}
        <ellipse cx="140" cy="105" rx="85" ry="65" fill="#334155" opacity="0.6" />
        {/* Spine */}
        <ellipse cx="140" cy="105" rx="12" ry="20" fill="#475569" />
        {/* Left kidney */}
        <path d="M 90 85 C 80 85 72 95 72 105 C 72 118 80 130 90 130 C 96 130 100 126 100 120 C 100 115 97 113 96 108 C 95 102 96 97 100 95 C 100 89 96 85 90 85Z" fill="#4A5568" />
        {/* Right kidney */}
        <path d="M 190 85 C 200 85 208 95 208 105 C 208 118 200 130 190 130 C 184 130 180 126 180 120 C 180 115 183 113 184 108 C 185 102 184 97 180 95 C 180 89 184 85 190 85Z" fill="#4A5568" />
        {/* Kidney interiors */}
        <path d="M 90 90 C 82 90 76 99 76 108 C 76 118 82 127 90 127 C 94 127 97 124 97 119 C 97 114 95 112 94 108 C 93 103 94 98 97 96 C 97 92 94 90 90 90Z" fill="#60A5FA" opacity="0.3" />
        <path d="M 190 90 C 198 90 204 99 204 108 C 204 118 198 127 190 127 C 186 127 183 124 183 119 C 183 114 185 112 186 108 C 187 103 186 98 183 96 C 183 92 186 90 190 90Z" fill="#60A5FA" opacity="0.3" />

        {/* Tumor */}
        {hasTumor && (
          <g>
            <circle cx={tumorX} cy={tumorY + 15} r={tumorR + 3} fill="#CBD5E1" opacity="0.25" />
            <circle cx={tumorX} cy={tumorY + 15} r={tumorR} fill="#94A3B8" opacity="0.75" />
            <circle cx={tumorX - 2} cy={tumorY + 13} r={tumorR * 0.4} fill="#E2E8F0" opacity="0.4" />
          </g>
        )}

        {/* Scan lines */}
        {[...Array(8)].map((_, i) => (
          <line key={i} x1="20" y1={30 + i * 22} x2="260" y2={30 + i * 22} stroke="#1E40AF" strokeWidth="0.3" opacity="0.2" />
        ))}

        {/* Labels */}
        <text x="10" y="18" fill="#64748B" fontSize="8" fontFamily="DM Mono, monospace">CT AXIAL</text>
        <text x="230" y="18" fill="#64748B" fontSize="8" fontFamily="DM Mono, monospace">R→L</text>
        <text x="72" y="145" fill="#94A3B8" fontSize="7" fontFamily="Inter, sans-serif">L Kidney</text>
        <text x="178" y="145" fill="#94A3B8" fontSize="7" fontFamily="Inter, sans-serif">R Kidney</text>
      </svg>
      {analysing && <div className="ct-scan-line absolute left-0 right-0 h-0.5 bg-cyan-300 shadow-[0_0_12px_rgba(103,232,249,0.9)]" />}
      <div className="absolute bottom-2 left-3 flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-sky-400"></div>
        <span className="text-xs text-slate-300 font-mono">Axial slice · Patient {String(patientId).padStart(3, '0')}</span>
      </div>
    </div>
  )
}

// ── Screens 6 & 7 — One stable detection interface ────────────
type DetectionState = 'ready' | 'analysing' | 'result'

const MODE_DETECTION_STAGES: Record<Mode, string[]> = {
  diabetes: [
    'Reading blood test values…',
    'Normalising numerical features…',
    'Feeding features into linear classifier…',
    'Calculating classification score…',
    'Generating binary prediction…',
  ],
  kidney: [
    'Loading CT scan…',
    'Preprocessing CT image…',
    'Extracting visual features…',
    'Running computer vision model…',
    'Calculating tumour probability…',
    'Generating prediction…',
  ],
}

interface FeatureContribution {
  feature: string
  percentage: number
  direction: 'positive' | 'negative'
}

function getDiabetesContributions(patient: Patient): FeatureContribution[] {
  const values = getBloodTests(patient)
  const modelFeatures = [
    { feature: 'Fasting Blood Glucose', reference: 100, scale: 25, coefficient: 1.4 },
    { feature: 'HbA1c', reference: 5.7, scale: 1, coefficient: 1.7 },
    { feature: 'Insulin', reference: 18, scale: 8, coefficient: 0.45 },
    { feature: 'Cholesterol', reference: 190, scale: 35, coefficient: 0.25 },
    { feature: 'Triglycerides', reference: 140, scale: 45, coefficient: 0.4 },
    { feature: 'HDL', reference: 50, scale: 12, coefficient: -0.6 },
    { feature: 'LDL', reference: 110, scale: 30, coefficient: 0.45 },
  ]
  const calculated = modelFeatures.map(item => {
    const patientValue = Number.parseFloat(values[item.feature as keyof typeof values])
    const signedInfluence = item.coefficient * ((patientValue - item.reference) / item.scale)
    return { ...item, signedInfluence, magnitude: Math.abs(signedInfluence) }
  })
  const totalMagnitude = calculated.reduce((sum, item) => sum + item.magnitude, 0)
  const contributions = calculated.map(item => ({
    feature: item.feature,
    percentage: Math.round((item.magnitude / totalMagnitude) * 100),
    direction: (item.signedInfluence >= 0 ? 'positive' : 'negative') as 'positive' | 'negative',
  }))
  const roundingDifference = 100 - contributions.reduce((sum, item) => sum + item.percentage, 0)
  const strongestIndex = calculated.reduce(
    (strongest, item, index, items) => item.magnitude > items[strongest].magnitude ? index : strongest,
    0,
  )
  contributions[strongestIndex].percentage += roundingDifference
  return contributions.sort((a, b) => b.percentage - a.percentage)
}

function DiabetesExplainability({ patient }: { patient: Patient }) {
  const contributions = getDiabetesContributions(patient)
  const rankDescriptions = ['Strongest influence', 'Second strongest influence', 'Third strongest influence']
  return (
    <>
      <div className="mb-5">
        <h4 className="text-sm font-bold text-slate-800 mb-3">Top Factors in This Prediction</h4>
        <div className="grid grid-cols-3 gap-2">
          {contributions.slice(0, 3).map((item, index) => (
            <div key={item.feature} className="rounded-xl border border-blue-100 bg-blue-50 p-3">
              <div className="w-6 h-6 rounded-full bg-white border border-blue-200 text-blue-700 flex items-center justify-center text-xs font-bold mb-2">#{index + 1}</div>
              <div className="text-xs font-bold text-slate-800 leading-tight min-h-8">{item.feature}</div>
              <div className="text-2xl font-bold text-blue-600 mt-1">{item.percentage}%</div>
              <div className="text-[10px] text-slate-500 mt-1">{rankDescriptions[index]}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="mb-5">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-bold text-slate-800">Relative Feature Contribution</h4>
          <span className="text-[10px] text-slate-400">Total: 100%</span>
        </div>
        <div className="space-y-3">
          {contributions.map(item => {
            const towardsDiabetes = item.direction === 'positive'
            return (
              <div key={item.feature}>
                <div className="flex items-start justify-between gap-3 mb-1">
                  <div>
                    <div className="text-xs font-semibold text-slate-700">{item.feature}</div>
                    <div className={`text-[10px] font-medium ${towardsDiabetes ? 'text-red-500' : 'text-blue-600'}`}>
                      {towardsDiabetes ? '↑ Towards Diabetes' : '↓ Towards No Diabetes'}
                    </div>
                  </div>
                  <span className="text-sm font-bold text-slate-800">{item.percentage}%</span>
                </div>
                <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${towardsDiabetes ? 'bg-red-500' : 'bg-blue-600'}`}
                    style={{ width: `${item.percentage}%` }}
                  />
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </>
  )
}

function KidneyExplainability({ patient }: { patient: Patient }) {
  const primaryAttention = 40 + (patient.id * 7) % 13
  const rightKidney = patient.id % 2 === 0
  const primaryX = rightKidney ? '68%' : '32%'
  return (
    <div className="mb-5">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h4 className="text-sm font-bold text-slate-800">CT Image Attention Map</h4>
        <span className="bg-violet-50 border border-violet-100 text-violet-700 rounded-full px-2 py-1 text-[10px] font-semibold">Educational AI visualization</span>
      </div>
      <div className="relative rounded-xl overflow-hidden">
        <KidneyCTScan patientId={patient.id} />
        <div className="pointer-events-none absolute inset-0" aria-hidden="true">
          <div className="absolute w-24 h-24 rounded-full bg-red-500/45 blur-xl" style={{ left: primaryX, top: '48%', transform: 'translate(-50%, -50%)' }} />
          <div className="absolute w-16 h-16 rounded-full bg-orange-400/40 blur-lg" style={{ left: rightKidney ? '58%' : '42%', top: '55%', transform: 'translate(-50%, -50%)' }} />
          <div className="absolute w-12 h-12 rounded-full bg-yellow-300/35 blur-md" style={{ left: rightKidney ? '76%' : '24%', top: '38%', transform: 'translate(-50%, -50%)' }} />
        </div>
      </div>
      <div className="flex items-center gap-3 mt-3">
        <span className="text-[10px] text-slate-500">Low influence</span>
        <div className="h-2 flex-1 rounded-full bg-gradient-to-r from-blue-400 via-yellow-300 to-red-500" />
        <span className="text-[10px] text-slate-500">High influence</span>
      </div>
      <div className="mt-3 rounded-lg bg-slate-50 border border-slate-100 px-3 py-2">
        <div className="text-[10px] uppercase tracking-wider font-semibold text-slate-400">Most Influential Region</div>
        <div className="text-sm font-semibold text-slate-800 mt-0.5">{rightKidney ? 'Right' : 'Left'} Kidney Region – {primaryAttention}% of highlighted model attention</div>
      </div>
      <p className="text-xs text-slate-500 leading-relaxed mt-3">The colours show where this demonstration model focused most strongly. A highlighted region is not necessarily a tumour or a medically validated lesion.</p>
    </div>
  )
}

function PredictionExplainability({ patient, mode }: { patient: Patient; mode: Mode }) {
  return (
    <section className="rounded-xl border border-slate-200 p-5 mb-5">
      <span className="inline-flex bg-blue-50 border border-blue-100 text-blue-700 rounded-full px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider mb-3">AI Decision Explanation</span>
      <h3 className="text-lg font-bold text-slate-900 mb-1">What Influenced This Prediction?</h3>
      <p className="text-xs text-slate-500 leading-relaxed mb-5">
        {mode === 'diabetes'
          ? "See which patient data had the strongest influence on the AI's decision."
          : "See which parts of the CT image had the strongest influence on the AI's decision."}
      </p>
      {mode === 'diabetes' ? <DiabetesExplainability patient={patient} /> : <KidneyExplainability patient={patient} />}
      <div className="bg-blue-50 border border-blue-100 rounded-xl p-4 text-xs text-blue-900 leading-relaxed">
        <div className="font-bold mb-1">How should I read this?</div>
        <p>{mode === 'diabetes'
          ? 'AI does not make its decision from just one blood test. It combines information from several features. A higher contribution percentage means that this feature had a stronger influence on this particular prediction.'
          : 'The attention map shows which image regions had more influence on this particular prediction. Warmer colours mean the model paid more attention to that area.'}</p>
        <p className="mt-2 font-medium">Contribution is not the same as medical importance. It only explains how this AI model used the data to make this prediction.</p>
      </div>
    </section>
  )
}

function Screen5({ patient, mode, onTestAnother, onRestart, onBack }: {
  patient: Patient
  mode: Mode
  onTestAnother: () => void
  onRestart: () => void
  onBack: () => void
}) {
  const [detectionState, setDetectionState] = useState<DetectionState>('ready')
  const [stage, setStage] = useState(-1)
  const stages = MODE_DETECTION_STAGES[mode]
  const positive = hasPrediction(patient, mode)
  const confidence = getConfidence(patient)
  const resultLabel = mode === 'diabetes'
    ? (positive ? 'Diabetes Detected' : 'No Diabetes Detected')
    : (positive ? 'Kidney Tumour Detected' : 'No Kidney Tumour Detected')

  const handleDetect = () => {
    setDetectionState('analysing')
    setStage(0)
  }

  useEffect(() => {
    if (detectionState !== 'analysing') return
    const timer = window.setInterval(() => {
      setStage(current => {
        if (current >= stages.length - 1) {
          window.clearInterval(timer)
          return stages.length
        }
        return current + 1
      })
    }, 650)
    return () => window.clearInterval(timer)
  }, [detectionState, stages.length])

  useEffect(() => {
    if (detectionState !== 'analysing' || stage !== stages.length) return
    const timer = window.setTimeout(() => setDetectionState('result'), 450)
    return () => window.clearTimeout(timer)
  }, [detectionState, stage, stages.length])

  return (
    <div className="max-w-7xl mx-auto px-6 py-6 animate-fade-in">
      <ProgressSteps current={detectionState === 'result' ? 7 : 6} />
      <div className="mb-4"><BackButton onClick={onBack} /></div>
      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,3fr)_minmax(22rem,2fr)] gap-8 items-start">
        <section className={`min-w-0 ${detectionState === 'result' ? 'order-2 lg:order-1' : 'order-1'}`}>
          <div className="flex items-center gap-2 mb-4">
            <h2 style={{ fontFamily: 'Outfit, sans-serif' }} className="text-xl font-bold text-slate-900">Patient Data</h2>
            <span className="text-xs font-medium bg-red-50 text-red-600 border border-red-100 px-2 py-0.5 rounded-full">
              {mode === 'diabetes' ? 'Diabetes Detection' : 'Kidney Tumour Detection'}
            </span>
          </div>
          <PatientDataPanel patient={patient} mode={mode} analysing={detectionState === 'analysing'} />
          {detectionState === 'result' && (
            <div className="mt-5">
              <PredictionExplainability patient={patient} mode={mode} />
            </div>
          )}
        </section>

        <aside className={`min-w-0 lg:sticky lg:top-6 ${detectionState === 'result' ? 'order-1 lg:order-2' : 'order-2'}`}>
          <div className="mb-4 flex items-center justify-between">
            <h2 style={{ fontFamily: 'Outfit, sans-serif' }} className="text-xl font-bold text-slate-900">
              {detectionState === 'result' ? 'AI Prediction Result' : detectionState === 'analysing' ? 'AI Analysing' : 'AI Detection'}
            </h2>
            <span className="text-xs font-mono text-slate-400">
              {mode === 'diabetes' ? 'LINEAR MODEL' : 'VISION MODEL'}
            </span>
          </div>
          <div className={`bg-white rounded-2xl border border-slate-200 shadow-sm p-6 flex flex-col transition-all duration-300 ${detectionState === 'result' ? 'lg:min-h-[31rem]' : 'min-h-[31rem]'}`}>
            {detectionState === 'ready' && (
              <div className="flex flex-col h-full animate-fade-in">
                <div className="bg-gradient-to-br from-blue-50 to-slate-50 rounded-2xl border border-blue-100 p-7 text-center mb-5">
                  <div className="w-20 h-20 rounded-full bg-white border border-blue-200 shadow-sm flex items-center justify-center mx-auto mb-4"><AIReadyIcon /></div>
                  <h3 className="text-lg font-bold text-slate-900 mb-2">Your {mode === 'diabetes' ? 'linear classification' : 'computer vision'} model is ready.</h3>
                  <p className="text-sm text-slate-500 leading-relaxed">
                    {mode === 'diabetes'
                      ? "The model will use only this patient's seven numerical blood-test features."
                      : "The model will use only this patient's kidney CT scan image."}
                  </p>
                </div>
                <div className="rounded-xl border border-slate-200 divide-y divide-slate-100 mb-5">
                  {(mode === 'diabetes'
                    ? [['Input', 'Blood Test Data'], ['Processing', 'Numerical Features'], ['Output', 'Diabetes / No Diabetes']]
                    : [['Input', 'Kidney CT Scan'], ['Processing', 'Image Features'], ['Output', 'Tumour / No Tumour']]
                  ).map(([label, value]) => (
                    <div key={label} className="flex justify-between gap-4 px-4 py-3 text-sm">
                      <span className="text-slate-400">{label}</span><span className="font-medium text-slate-700 text-right">{value}</span>
                    </div>
                  ))}
                </div>
                <button onClick={handleDetect} className="mt-auto w-full bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 rounded-xl font-bold shadow-lg transition-all active:scale-95">
                  Start AI Detection
                </button>
              </div>
            )}

            {detectionState === 'analysing' && (
              <div className="animate-fade-in">
                <div className="text-center mb-6">
                  <div className="relative w-20 h-20 mx-auto mb-4 flex items-center justify-center">
                    <div className="absolute inset-0 rounded-full border-4 border-blue-100" />
                    <div className="absolute inset-0 rounded-full border-4 border-blue-600 border-t-transparent animate-spin" />
                    <AIReadyIcon />
                  </div>
                  <div className="font-semibold text-blue-700">Analysing this patient's {mode === 'diabetes' ? 'blood tests' : 'CT scan'}…</div>
                  <div className="text-xs text-slate-400 mt-1">Each step uses the data shown on the left.</div>
                </div>
                <div className="border border-slate-200 rounded-xl overflow-hidden divide-y divide-slate-100">
                  {stages.map((label, index) => (
                    <div key={label} className={`flex items-center gap-3 px-4 py-3 transition-colors ${index < stage ? 'bg-emerald-50' : index === stage ? 'bg-blue-50' : 'bg-white'}`}>
                      {index < stage ? (
                        <span className="w-5 h-5 rounded-full bg-emerald-500 text-white flex items-center justify-center text-xs">✓</span>
                      ) : index === stage ? (
                        <span className="w-5 h-5 rounded-full border-2 border-blue-600 border-t-transparent animate-spin" />
                      ) : (
                        <span className="w-5 h-5 rounded-full border-2 border-slate-200" />
                      )}
                      <span className={`text-sm font-medium ${index < stage ? 'text-emerald-700' : index === stage ? 'text-blue-700' : 'text-slate-400'}`}>{label}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {detectionState === 'result' && (
              <div className="flex flex-col h-full animate-scale-in">
                <div className={`rounded-2xl border-2 p-6 text-center ${positive ? 'bg-red-50 border-red-200' : 'bg-emerald-50 border-emerald-200'}`}>
                  <div className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">Predicted Outcome</div>
                  <div className={`text-2xl font-extrabold ${positive ? 'text-red-700' : 'text-emerald-700'}`}>{resultLabel}</div>
                  <div className="text-sm text-slate-500 mt-2">Binary classification result</div>
                </div>
                <div className="my-5 rounded-xl border border-slate-200 p-5">
                  <div className="flex items-end justify-between mb-3">
                    <div>
                      <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Prediction Confidence</div>
                      <div className="text-xs text-slate-400 mt-1">Demonstration model output</div>
                    </div>
                    <div className="text-3xl font-bold text-blue-600">{confidence}%</div>
                  </div>
                  <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden"><div className="h-full bg-blue-600 rounded-full confidence-bar" style={{ width: `${confidence}%` }} /></div>
                </div>
                <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-xs text-amber-800 leading-relaxed mb-5">
                  <strong>Educational prototype:</strong> This result was generated from {mode === 'diabetes' ? 'blood-test numerical features using a linear binary classifier' : 'a kidney CT image using a computer vision classifier'}. This is an educational prototype and is not a medical diagnosis.
                </div>
                <div className="mt-auto hidden lg:flex gap-3">
                  <button onClick={onTestAnother} className="flex-1 border-2 border-red-200 text-red-600 hover:bg-red-50 px-3 py-3 rounded-xl font-semibold text-sm transition-colors">Test Another</button>
                  <button onClick={onRestart} className="flex-1 bg-slate-800 hover:bg-slate-900 text-white px-3 py-3 rounded-xl font-semibold text-sm transition-colors">Start Over</button>
                </div>
              </div>
            )}
          </div>
        </aside>
      </div>
      {detectionState === 'result' && (
        <div className="flex lg:hidden gap-3 mt-6">
          <button onClick={onTestAnother} className="flex-1 border-2 border-red-200 text-red-600 hover:bg-red-50 px-3 py-3 rounded-xl font-semibold text-sm transition-colors">Test Another</button>
          <button onClick={onRestart} className="flex-1 bg-slate-800 hover:bg-slate-900 text-white px-3 py-3 rounded-xl font-semibold text-sm transition-colors">Start Over</button>
        </div>
      )}
    </div>
  )
}

function AIReadyIcon() {
  return (
    <svg width="44" height="44" viewBox="0 0 44 44" fill="none" className="animate-float">
      <path d="M22 6L26.5 17.5L38 22L26.5 26.5L22 38L17.5 26.5L6 22L17.5 17.5L22 6Z" fill="#3B82F6" opacity="0.8" />
      <circle cx="32" cy="12" r="3" fill="#60A5FA" opacity="0.6" />
      <circle cx="12" cy="32" r="2" fill="#93C5FD" opacity="0.5" />
    </svg>
  )
}

// ── Main App ───────────────────────────────────────────────────
export default function App() {
  const [screen, setScreen] = useState<ScreenId>(1)
  const [mode, setMode] = useState<Mode | null>(null)
  const [trainingIds, setTrainingIds] = useState<Set<number>>(new Set())
  const [testId, setTestId] = useState<number | null>(null)
  const [trainingComplete, setTrainingComplete] = useState(false)
  const [evaluationComplete, setEvaluationComplete] = useState(false)
  const [trainingLogCount, setTrainingLogCount] = useState(0)

  const resetWorkflow = () => {
    setScreen(1)
    setMode(null)
    setTrainingIds(new Set())
    setTestId(null)
    setTrainingComplete(false)
    setEvaluationComplete(false)
    setTrainingLogCount(0)
  }

  const handleSelectMode = (m: Mode) => {
    if (mode === m) {
      setScreen(2)
      return
    }
    setMode(m)
    setTestId(null)
    setTrainingComplete(false)
    setEvaluationComplete(false)
    setTrainingLogCount(0)
    setScreen(2)
  }

  const handleToggleTraining = (id: number) => {
    setTrainingIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else if (next.size < 20) next.add(id)
      return next
    })
    setTrainingComplete(false)
    setEvaluationComplete(false)
    setTrainingLogCount(0)
    setTestId(null)
  }

  const handleRandomTraining = () => {
    const shuffledIds = ALL_PATIENTS.map(patient => patient.id)
    for (let index = shuffledIds.length - 1; index > 0; index--) {
      const swapIndex = Math.floor(Math.random() * (index + 1))
      ;[shuffledIds[index], shuffledIds[swapIndex]] = [shuffledIds[swapIndex], shuffledIds[index]]
    }
    const repeatsCurrentSelection =
      trainingIds.size === 20 && shuffledIds.slice(0, 20).every(id => trainingIds.has(id))
    if (repeatsCurrentSelection) {
      ;[shuffledIds[0], shuffledIds[20]] = [shuffledIds[20], shuffledIds[0]]
    }
    setTrainingIds(new Set(shuffledIds.slice(0, 20)))
    setTrainingComplete(false)
    setEvaluationComplete(false)
    setTrainingLogCount(0)
    setTestId(null)
  }

  const handleSelectTest = (id: number) => {
    setTestId(id)
  }

  const evaluationPatients = ALL_PATIENTS.filter(patient => !trainingIds.has(patient.id))
  const testPatient = testId ? NEW_PATIENTS.find(patient => patient.id === testId) ?? null : null

  return (
    <div className="min-h-screen" style={{ background: '#F8FAFC' }}>
      <Header onHome={() => setScreen(1)} />
      {screen === 1 && <Screen1 onSelect={handleSelectMode} />}
      {screen === 2 && (
        <Screen2
          trainingIds={trainingIds}
          mode={mode!}
          onToggle={handleToggleTraining}
          onRandomSelect={handleRandomTraining}
          onNext={() => setScreen(3)}
          onBack={() => setScreen(1)}
        />
      )}
      {screen === 3 && mode && (
        <Screen3
          mode={mode}
          completed={trainingComplete}
          initialVisibleCount={trainingLogCount}
          onProgress={setTrainingLogCount}
          onTrainingComplete={() => setTrainingComplete(true)}
          onContinue={() => setScreen(4)}
          onBack={() => setScreen(2)}
        />
      )}
      {screen === 4 && mode && (
        <EvaluationDataScreen
          patients={evaluationPatients}
          mode={mode}
          completed={evaluationComplete}
          onComplete={() => { setEvaluationComplete(true); setScreen(5) }}
          onBack={() => setScreen(3)}
        />
      )}
      {screen === 5 && mode && (
        <EvaluationResultsScreen
          patients={evaluationPatients}
          mode={mode}
          onNext={() => setScreen(6)}
          onBack={() => setScreen(4)}
        />
      )}
      {screen === 6 && mode && (
        <SelectTestPatientScreen
          mode={mode}
          testId={testId}
          onSelect={handleSelectTest}
          onNext={() => setScreen(7)}
          onBack={() => setScreen(5)}
        />
      )}
      {screen === 7 && testPatient && mode && (
        <Screen5
          patient={testPatient}
          mode={mode}
          onTestAnother={() => { setTestId(null); setScreen(6) }}
          onRestart={resetWorkflow}
          onBack={() => setScreen(6)}
        />
      )}
    </div>
  )
}
