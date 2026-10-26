import { useState, useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import { HOME_ROUTE, pathForRoute, routeFromPath, type AppRoute, type Mode, type ScreenId } from './routes'

import { usePatients, getBloodTests, type Patient } from './patients'
import KidneyCTViewer from './KidneyCTViewer'
import {
  evaluationPatientCodes,
  isKidneyPrediction,
  isKidneySession,
  ModelApiError,
  overlayUrl,
  predictPatient,
  trainModel,
  type AnyPrediction,
  type KidneyPrediction,
  type KidneySession,
  type ModelSession,
  type PredictionResult,
  type TrainingPoint,
} from './api'

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
  const positive = mode === 'diabetes' && patient.label === 'diabetes_positive'
  const label = mode === 'diabetes'
    ? (positive ? 'Diabetes' : 'No Diabetes')
    : `${patient.slice_count} CT slices · ${patient.pool}`
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
          <p className="text-slate-500 text-sm leading-relaxed mb-5">Learn how a Random Forest uses six clinical features to predict synthetic diabetes labels.</p>
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
function Screen2({ patients, trainingIds, mode, onToggle, onRandomSelect, onNext, onBack }: {
  patients: Patient[]
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
          {mode === 'kidney'
            ? 'Browse all 40 CT cases. The manifest fixes 20 training, 10 validation and 10 test patients. Only training cases can be selected.'
            : 'Browse all 40 teaching patients. Select 20 of the first 30 for training; the remaining 10 are used for evaluation. P031–P040 are held out for prediction.'}
        </div>
        <div className="mt-4 flex flex-wrap items-center justify-center gap-3 text-sm font-semibold">
          <span className="bg-slate-100 text-slate-700 px-4 py-2 rounded-lg">40 Dataset Patients</span>
          <span className="text-slate-400">→</span>
          <span className="bg-blue-50 border border-blue-200 text-blue-700 px-4 py-2 rounded-lg">20 Training Patients</span>
          <span className="text-slate-400">+</span>
          <span className="bg-orange-50 border border-orange-200 text-orange-700 px-4 py-2 rounded-lg">10 Evaluation Patients + 10 Test Patients</span>
        </div>
        <button
          type="button"
          onClick={onRandomSelect}
          className="mt-4 inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-5 py-2.5 rounded-xl text-sm font-semibold shadow-md transition-colors"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M2 4h2.2c3.8 0 3.8 8 7.6 8H14M11 9l3 3-3 3M2 12h2.2c1.2 0 2-.8 2.7-1.9M9 5.9C9.7 4.8 10.5 4 11.8 4H14M11 1l3 3-3 3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          {mode === 'kidney' ? 'Select 20 Training Cases' : 'Randomly Select 20 Patients'}
        </button>
      </div>

      {/* Patient grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 mb-28">
        {patients.map(p => {
          const selected = trainingIds.has(p.id)
          const eligible = mode === 'kidney' ? p.pool === 'train' : p.pool === 'development'
          const disabled = !eligible || (!selected && count >= MAX)
          return (
            <div
              key={p.id}
              className={`relative bg-white rounded-xl border-2 p-3 pb-9 flex flex-col items-center gap-2 transition-all duration-200 focus:outline-none ${
                selected ? 'patient-card-selected-train' :
                disabled ? 'cursor-default border-slate-200' :
                'border-slate-200 hover:border-blue-200 hover:shadow-md hover:-translate-y-0.5 cursor-pointer'
              }`}
            >
              {selected && (
                <div className="absolute top-1.5 right-1.5 w-5 h-5 bg-blue-600 rounded-full flex items-center justify-center checkmark-anim z-10">
                  <svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M 1.5 5 L 4 7.5 L 8.5 2.5" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
                </div>
              )}
              <button type="button" disabled={disabled} aria-pressed={selected}
                aria-label={`Select ${p.name} for training`}
                onClick={() => onToggle(p.id)}
                className="flex w-full flex-col items-center gap-2 rounded-lg focus-visible:outline-2 focus-visible:outline-blue-600 disabled:cursor-default">
              <PatientAvatar id={p.id} size={52} />
              <div className="text-center">
                <div className="text-xs font-semibold text-slate-800 leading-tight">{p.name}</div>
                <div className="text-xs text-slate-400 mt-0.5" style={{ fontFamily: 'DM Mono, monospace' }}>{mode === 'diabetes' ? `Age ${p.age}` : p.case_id}</div>
                <DiseaseStatusBadge patient={p} mode={mode} />
              </div>
              </button>
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
function formatDice(value: number | null | undefined) {
  return value === null || value === undefined ? 'n/a' : value.toFixed(4)
}

function makeStartupLogs(mode: Mode, session: ModelSession | null, selectedCount: number) {
  if (!session) {
    return [
      'Connecting to the local model service…',
      `Sending ${selectedCount} selected ${mode === 'kidney' ? 'CT cases' : 'training patients'}…`,
      'Waiting for the model on the server…',
    ]
  }
  if (isKidneySession(session)) {
    const selection = session.selection
    return [
      'Loading the reviewed PNG CT handoff…',
      `Training cases: ${selection.training_size} · held-out cases: ${selection.evaluation_size}`,
      `Training slices: ${selection.training_slice_count ?? '—'} · held-out slices: ${selection.evaluation_slice_count ?? '—'}`,
      `Building a 3-class U-Net (${(session.model.class_names ?? []).join(' / ')}) at ${session.model.input_size}×${session.model.input_size}`,
      `Device: ${session.model.device} · ${session.model.epochs} epochs · replaying the shipped checkpoint's recorded run`,
    ]
  }
  const points = session.training_curve.points
  const maxTrees = points.length ? points[points.length - 1].trees_built : 0
  return [
    'Loading the 40-patient synthetic diabetes cohort…',
    `Validated ${session.selection.training_size} training patients.`,
    `Reserved ${session.selection.evaluation_size} evaluation patients: ${session.split.evaluation.join(', ')}`,
    `Held out ${session.selection.held_out_size} prediction patients: ${session.split.held_out.join(', ')}`,
    `Features: ${session.model.feature_names.join(', ')}`,
    'Building trees with class_weight=balanced, max_depth=3, min_samples_leaf=2',
    `Tree checkpoints from 5 to ${maxTrees}`,
  ]
}

function makeTrainingLogs(mode: Mode, session: ModelSession | null, selectedCount: number) {
  const startup = makeStartupLogs(mode, session, selectedCount)
  if (!session) return startup

  if (isKidneySession(session)) {
    const checkpoints = session.training_curve.points.map(point =>
      `Epoch ${point.epoch}/${session.model.epochs}  loss: ${point.train_loss.toFixed(4)} - kidney Dice: ${formatDice(point.validation_kidney_dice)} - tumour Dice: ${formatDice(point.validation_tumour_dice)}`,
    )
    const metrics = session.evaluation.metrics
    return [
      ...startup,
      ...checkpoints,
      'Selecting the epoch with the best validation tumour Dice…',
      `✓ Training complete — epoch ${session.model.selected_epoch} kept`,
      `✓ Held-out test: kidney Dice ${formatDice(metrics.kidney_dice)} · tumour Dice ${formatDice(metrics.tumour_dice)}`,
    ]
  }

  const points = session.training_curve.points
  const maxTrees = points.length ? points[points.length - 1].trees_built : 0
  const checkpoints = points.map((point: TrainingPoint) =>
    `Trees ${point.trees_built}/${maxTrees}  training accuracy: ${point.train_accuracy.toFixed(4)} - evaluation accuracy: ${point.validation_accuracy.toFixed(4)}`,
  )
  return [
    ...startup,
    ...checkpoints,
    'Selecting the smallest tree count with the best evaluation accuracy…',
    `✓ Training complete — ${session.model.best_tree_count} trees selected`,
    `✓ Evaluation accuracy: ${(session.model.best_evaluation_accuracy * 100).toFixed(1)}%`,
  ]
}

const TRAINING_CONCEPTS = [
  {
    term: 'Tree',
    title: 'Tree – One Pattern-Finding Decision Maker',
    explanation: 'This model is a Random Forest, so it is built from many small decision trees. Each new tree is added and then the model is checked again on the training and evaluation patients.',
    example: 'Trees 15/100 means the forest currently contains 15 decision trees.',
  },
  {
    term: 'Model Selection',
    title: 'Model Selection – How Many Trees?',
    explanation: 'After each checkpoint the evaluation accuracy is recorded. The smallest forest size that reaches the best evaluation accuracy is kept, so the model is chosen by testing, not by guessing.',
    example: 'If 45 trees score best on the evaluation patients, the model keeps 45 trees.',
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

function Screen3({ mode, session, loading, error, unavailable, selectedCount, completed, initialVisibleCount, onProgress, onTrainingComplete, onContinue, onBack, onRetry }: {
  mode: Mode
  session: ModelSession | null
  loading: boolean
  error: string
  unavailable: boolean
  selectedCount: number
  completed: boolean
  initialVisibleCount: number
  onProgress: (count: number) => void
  onTrainingComplete: () => void
  onContinue: () => void
  onBack: () => void
  onRetry: () => void
}) {
  const logs = makeTrainingLogs(mode, session, selectedCount)
  const [visibleCount, setVisibleCount] = useState(completed ? logs.length : initialVisibleCount)
  const consoleRef = useRef<HTMLDivElement>(null)
  const done = !loading && !error && session !== null && visibleCount >= logs.length
  const canContinue = done || unavailable

  useEffect(() => {
    if (completed) {
      setVisibleCount(logs.length)
      return
    }
    if (!logs.length) return
    const timer = window.setInterval(() => {
      setVisibleCount(count => (count >= logs.length ? count : count + 1))
    }, 180)
    return () => window.clearInterval(timer)
  }, [mode, logs.length, completed])

  useEffect(() => {
    if (completed) return
    onProgress(visibleCount)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visibleCount, completed])

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
        <p className="text-slate-500">{mode === 'kidney' ? 'Replay the included U-Net’s recorded training history. No retraining runs in this page.' : 'Review the measured Random Forest checkpoints returned by the backend.'}</p>
      </div>

      <div className={`border rounded-xl px-5 py-4 mb-5 text-sm ${mode === 'diabetes' ? 'bg-blue-50 border-blue-100 text-blue-800' : 'bg-slate-100 border-slate-200 text-slate-700'}`}>
        {mode === 'kidney' ? 'The existing U-Net was fitted on the locked 20 training cases. Ten validation cases selected its epoch; the ten final-test cases were separate.' : 'The Random Forest fits 20 selected patients and uses 10 validation patients to choose its tree count. P031–P040 are kept for final testing.'}
      </div>

      {error && (
        <div className={`rounded-xl border px-5 py-4 mb-5 text-sm ${unavailable ? 'bg-amber-50 border-amber-200 text-amber-900' : 'bg-red-50 border-red-200 text-red-800'}`} role="alert">
          <div className="font-semibold mb-1">
            {unavailable ? 'The model service is not connected for this module' : 'Model training failed'}
          </div>
          <p className="leading-relaxed">{error}</p>
          {!unavailable && <button type="button" onClick={onRetry} className="mt-3 bg-red-600 text-white rounded-lg px-4 py-2 text-xs font-semibold">Retry training</button>}
        </div>
      )}

      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden mb-5">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-px bg-slate-200 border-b border-slate-200">
          {[
            ['Model', mode === 'diabetes' ? 'Random Forest Classifier' : 'U-Net Segmentation'],
            ['Input', mode === 'diabetes' ? 'Blood Test Features' : 'Kidney CT Images'],
            ['Training Dataset', `${session?.selection.training_size ?? 20} patients`],
            ['Evaluation Dataset', `${session?.selection.evaluation_size ?? 10} reserved · unseen`],
            mode === 'diabetes'
              ? ['Selected Trees', session && !isKidneySession(session) ? String(session.model.best_tree_count) : '—']
              : ['Selected Epoch', session && isKidneySession(session) ? String(session.model.selected_epoch) : '—'],
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
                line.startsWith('Trees ') ? 'text-sky-300' :
                line.startsWith('Features:') || line.startsWith('Tree checkpoints') ? 'text-amber-300' :
                'text-slate-300'
              }
            >
              <span className="text-slate-600 select-none mr-3">$</span>{line}
            </div>
          ))}
          {!done && <span className="terminal-cursor inline-block w-2 h-4 bg-emerald-400 align-middle" />}
        </div>
      </div>

      {mode === 'diabetes'
        ? <TrainingEducation />
        : (
          <section className="mb-5 bg-white rounded-xl border border-slate-200 shadow-sm p-5">
            <h2 className="text-lg font-bold text-slate-900 mb-2">How the kidney model learns</h2>
            <p className="text-sm text-slate-600 leading-relaxed">
              This activity uses an already trained U-Net, a small neural network that labels every pixel of a CT slice as
              background, kidney or tumour. During offline training, each epoch was scored on ten separate validation cases with{' '}
              <strong>Dice</strong> — the overlap between the predicted region and the reference region, where 1.0 is a
              perfect match. The epoch with the best tumour Dice on the validation cases is kept.
            </p>
            <p className="text-xs text-slate-500 mt-3">
              All ten held-out cases contain a tumour, so "detected / not detected" is not a useful score here — the Dice
              values are the meaningful measure. The teaching checkpoint is included. Optional retraining uses{' '}
              <span className="font-mono">python backend/train_kidney_model.py</span>.
            </p>
          </section>
        )}

      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <span className={`w-3 h-3 rounded-full ${done ? 'bg-emerald-500' : error ? 'bg-amber-500' : 'bg-blue-500 animate-pulse'}`} />
          <div>
            <div className={`text-sm font-semibold ${done ? 'text-emerald-700' : error ? 'text-amber-700' : 'text-blue-700'}`}>
              {done ? (mode === 'kidney' ? 'Recorded Training Loaded' : 'AI Model Training Complete') : error ? 'Model results unavailable' : (mode === 'kidney' ? 'Loading Recorded Training…' : 'Training in progress…')}
            </div>
            <div className="text-xs text-slate-400">
              {done
                ? (isKidneySession(session)
                  ? `Loaded recorded metrics — U-Net epoch ${session.model.selected_epoch} kept for evaluation.`
                  : `Trained on the server — ${session?.model.best_tree_count} trees kept for evaluation.`)
                : error
                  ? 'You can still browse the stored patient dataset.'
                  : (mode === 'kidney'
                    ? 'The backend is loading the U-Net checkpoint for this activity.'
                    : 'The backend is fitting a Random Forest for your selection.')}
            </div>
          </div>
        </div>
        <button
          onClick={onContinue}
          disabled={!canContinue}
          className={`inline-flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-sm transition-all ${
            canContinue ? 'bg-blue-600 text-white hover:bg-blue-700 shadow-md' : 'bg-slate-100 text-slate-400 cursor-not-allowed'
          }`}
        >
          {done ? 'Continue to Evaluation' : 'Continue without a model'}
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
    ? ['Reviewing clinical features…', 'Reviewing Random Forest output…', 'Comparing known labels…', 'Displaying recorded metrics…']
    : ['Reviewing CT case identities…', 'Reading checkpoint evaluation…', 'Comparing reference masks…', 'Displaying recorded Dice…']

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
        <p className="text-slate-500">These 10 patients match the evaluation report shown in the next step.</p>
      </div>
      <div className="bg-orange-50 border border-orange-200 rounded-xl px-5 py-4 mb-6 text-sm text-orange-900 leading-relaxed">
        <strong>These patients were not used to fit the model.</strong> {mode === 'kidney' ? 'This screen reviews the checkpoint’s recorded held-out test report; it does not rerun testing.' : 'These validation patients helped select the tree count; their results are not an independent final-test score.'}
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
                  <div className="text-xs text-slate-400 mt-0.5 font-mono">{mode === 'diabetes' ? `Age ${patient.age}` : patient.case_id}</div>
                  <DiseaseStatusBadge patient={patient} mode={mode} />
                </div>
                <ViewDataButton patient={patient} onView={setViewingPatient} />
              </div>
            ))}
          </div>
          <div className="text-center">
            <button onClick={() => completed ? onComplete() : (setEvaluating(true), setCurrentPatient(0))} className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 rounded-xl font-bold shadow-lg transition-all active:scale-95">
              {completed ? 'View Evaluation Results' : 'Review Evaluation Results'}
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
            <h2 className="text-xl font-bold text-slate-900 mb-2">Reviewing recorded evaluation…</h2>
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
function KidneyEvaluationResults({ session, patients, onNext, onBack }: {
  session: KidneySession
  patients: Patient[]
  onNext: () => void
  onBack: () => void
}) {
  const metrics = session.evaluation.metrics
  const byCode = new Map(patients.map(patient => [patient.patient_code, patient]))
  const dice = (value: number | null) => (value === null ? 'n/a' : value.toFixed(3))
  const tumourCases = session.evaluation.predictions.filter(row => row.actual_label === 1).length

  return (
    <div className="max-w-6xl mx-auto px-6 py-6 animate-fade-in">
      <ProgressSteps current={4} />
      <div className="mb-4"><BackButton onClick={onBack} /></div>
      <div className="text-center mb-7">
        <h1 className="text-3xl font-bold text-slate-900 mb-2">Model Evaluation Results</h1>
        <p className="text-slate-500 max-w-2xl mx-auto">
          The U-Net was scored on {session.evaluation.sample_count} held-out CT cases that were never used for training.
        </p>
        <span className="inline-block mt-3 bg-emerald-50 text-emerald-700 border border-emerald-100 rounded-full px-3 py-1 text-xs font-medium">
          Recorded held-out evaluation for the included U-Net checkpoint
        </span>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-7">
        {[
          ['Held-out Cases', String(session.evaluation.sample_count), 'text-slate-800'],
          ['Kidney Dice', dice(metrics.kidney_dice), 'text-blue-600'],
          ['Tumour Dice', dice(metrics.tumour_dice), 'text-red-600'],
          ['Case Labels Matched', `${metrics.correct} / ${metrics.total}`, 'text-emerald-600'],
        ].map(([label, value, color]) => (
          <div key={label} className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 text-center">
            <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">{label}</div>
            <div className={`text-3xl font-bold ${color}`}>{value}</div>
          </div>
        ))}
      </div>

      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden mb-7">
        <div className="px-5 py-4 bg-slate-50 border-b border-slate-200">
          <h2 className="text-lg font-bold text-slate-900">Per-case overlap with the reference mask</h2>
        </div>
        <div className="divide-y divide-slate-100">
          {session.evaluation.predictions.map(row => {
            const patient = byCode.get(row.patient_id)
            return (
              <div key={row.case_id} className="grid grid-cols-[minmax(10rem,1.6fr)_1fr_1fr_1fr_auto] gap-4 items-center px-5 py-4">
                <div className="flex items-center gap-3 min-w-0">
                  <PatientAvatar id={patient?.id ?? 0} size={42} />
                  <div className="min-w-0">
                    <div className="font-semibold text-sm text-slate-800 truncate">{patient?.name ?? row.patient_id}</div>
                    <div className="text-[11px] text-slate-400 font-mono">{row.patient_id} · {row.case_id}</div>
                  </div>
                </div>
                <div>
                  <div className="text-xs text-slate-400">Reference</div>
                  <div className="text-sm font-medium text-slate-700">{row.actual_label === 1 ? 'Tumour' : 'No tumour'}</div>
                </div>
                <div>
                  <div className="text-xs text-slate-400">Model</div>
                  <div className="text-sm font-medium text-slate-700">{row.predicted_label === 1 ? 'Tumour' : 'No tumour'}</div>
                </div>
                <div>
                  <div className="text-xs text-slate-400">Dice (kidney / tumour)</div>
                  <div className="text-sm font-mono text-slate-700">{dice(row.kidney_dice)} / {dice(row.tumour_dice)}</div>
                </div>
                <span className={`min-w-24 text-center rounded-full px-3 py-1.5 text-xs font-bold ${row.correct ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-red-50 text-red-700 border border-red-200'}`}>
                  {row.correct ? '✓ Matched' : '✕ Different'}
                </span>
              </div>
            )
          })}
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-100 rounded-2xl p-6 mb-7">
        <h2 className="text-lg font-bold text-blue-900 mb-2">Reading these numbers</h2>
        <p className="text-sm text-blue-800 leading-relaxed">
          Dice measures how much the predicted kidney and tumour regions overlap the reference mask: 1.0 is a perfect
          match, 0.0 is no overlap. Kidney segmentation is much easier than tumour segmentation because tumours are small
          and vary in appearance, which is why the tumour score is lower.
        </p>
        <p className="mt-3 text-xs text-blue-800">
          All {tumourCases} held-out cases contain a tumour, so the {metrics.correct}/{metrics.total} label agreement is not a
          meaningful detector score. It is a 40-case teaching subset, not clinical validation.
        </p>
      </div>

      <div className="text-center">
        <button onClick={onNext} className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-9 py-3.5 rounded-xl font-bold shadow-md transition-colors">Next <span aria-hidden="true">→</span></button>
      </div>
    </div>
  )
}

function EvaluationResultsScreen({ patients, mode, session, onNext, onBack }: {
  patients: Patient[]
  mode: Mode
  session: ModelSession | null
  onNext: () => void
  onBack: () => void
}) {
  if (isKidneySession(session)) {
    return <KidneyEvaluationResults session={session} patients={patients} onNext={onNext} onBack={onBack} />
  }
  const evaluation = session?.evaluation ?? null
  const metrics = evaluation?.metrics ?? null
  const positiveLabel = mode === 'diabetes' ? 'Diabetes' : 'Kidney Tumour'
  const negativeLabel = mode === 'diabetes' ? 'No Diabetes' : 'No Kidney Tumour'
  const byCode = new Map(patients.map(patient => [patient.patient_code, patient]))
  const rows = evaluation
    ? evaluation.predictions.map(prediction => ({
        prediction,
        patient: byCode.get(prediction.patient_id) ?? null,
      }))
    : []
  const total = evaluation?.sample_count ?? 0
  const correct = evaluation ? evaluation.predictions.filter(row => row.correct).length : 0
  const accuracyText = metrics ? `${(metrics.accuracy * 100).toFixed(1)}%` : '—'

  if (!evaluation) {
    return (
      <div className="max-w-6xl mx-auto px-6 py-6 animate-fade-in">
        <ProgressSteps current={4} />
        <div className="mb-4"><BackButton onClick={onBack} /></div>
        <div className="text-center mb-7">
          <h1 className="text-3xl font-bold text-slate-900 mb-2">Model Evaluation Results</h1>
        </div>
        <div className="bg-amber-50 border border-amber-200 rounded-2xl p-6 max-w-2xl mx-auto text-sm text-amber-900" role="status">
          <p className="font-semibold mb-2">No model results are available for this module.</p>
          <p className="leading-relaxed">
            The {mode === 'kidney' ? 'kidney segmentation' : 'diabetes'} model did not run, so there are no predictions or metrics to report.
            Go back to the training step to read the reason, or continue to browse the stored patient dataset.
          </p>
        </div>
        <div className="text-center mt-7">
          <button onClick={onNext} className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-9 py-3.5 rounded-xl font-bold shadow-md transition-colors">Continue <span aria-hidden="true">→</span></button>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto px-6 py-6 animate-fade-in">
      <ProgressSteps current={4} />
      <div className="mb-4"><BackButton onClick={onBack} /></div>
      <div className="text-center mb-7">
        <h1 className="text-3xl font-bold text-slate-900 mb-2">Model Evaluation Results</h1>
        <p className="text-slate-500 max-w-2xl mx-auto">Your trained model has been evaluated using {total} unseen patients that were not included in the Training Dataset.</p>
        <span className="inline-block mt-3 bg-emerald-50 text-emerald-700 border border-emerald-100 rounded-full px-3 py-1 text-xs font-medium">Live output from the local Random Forest model</span>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-7">
        {[
          ['Evaluation Patients', String(total), 'text-slate-800'],
          ['Correct Predictions', `${correct} / ${total}`, 'text-emerald-600'],
          ['Incorrect Predictions', `${total - correct} / ${total}`, 'text-red-600'],
          ['Accuracy', accuracyText, 'text-blue-600'],
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
          {rows.map(({ patient, prediction }) => (
            <div key={prediction.patient_id} className="grid grid-cols-[minmax(10rem,1.4fr)_1fr_1fr_auto] gap-4 items-center px-5 py-4">
              <div className="flex items-center gap-3 min-w-0">
                <PatientAvatar id={patient?.id ?? 0} size={42} />
                <div className="min-w-0">
                  <div className="font-semibold text-sm text-slate-800 truncate">{patient?.name ?? prediction.patient_id}</div>
                  <div className="text-[11px] text-slate-400 font-mono">{prediction.patient_id}</div>
                </div>
              </div>
              <div><div className="text-xs text-slate-400">Actual</div><div className="text-sm font-medium text-slate-700">{prediction.actual_label === 1 ? positiveLabel : negativeLabel}</div></div>
              <div>
                <div className="text-xs text-slate-400">AI Prediction</div>
                <div className="text-sm font-medium text-slate-700">{prediction.predicted_label === 1 ? positiveLabel : negativeLabel}</div>
                <div className="text-[11px] text-slate-400 font-mono">score {(prediction.estimated_diabetes_probability * 100).toFixed(1)}%</div>
              </div>
              <span className={`min-w-24 text-center rounded-full px-3 py-1.5 text-xs font-bold ${prediction.correct ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-red-50 text-red-700 border border-red-200'}`}>{prediction.correct ? '✓ Correct' : '✕ Incorrect'}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-100 rounded-2xl p-6 mb-7">
        <h2 className="text-lg font-bold text-blue-900 mb-2">Why do we evaluate the model?</h2>
        <p className="text-sm text-blue-800 leading-relaxed mb-5">Evaluation Data helps us measure how well the trained model performs on patients it has never seen before. Comparing the AI predictions with the known correct labels gives us an estimate of the model’s performance.</p>
        <div className="flex flex-wrap items-center justify-center gap-2 text-xs font-semibold text-slate-700">
          <span className="bg-white border border-blue-200 rounded-lg px-3 py-2">{total} Unseen Patients</span><span>→</span>
          <span className="bg-white border border-blue-200 rounded-lg px-3 py-2">Trained AI Model</span><span>→</span>
          <span className="bg-white border border-blue-200 rounded-lg px-3 py-2">AI Predictions</span><span>+</span>
          <span className="bg-white border border-blue-200 rounded-lg px-3 py-2">Actual Labels</span><span>→</span>
          <span className="bg-blue-600 text-white rounded-lg px-3 py-2">Compare → {accuracyText} Accuracy</span>
        </div>
        <div className="text-center mt-4 font-mono text-sm text-blue-900">{correct} correct predictions ÷ {total} evaluation patients = {accuracyText} accuracy</div>
        <p className="mt-4 text-center text-xs text-blue-800">
          These {total} patients never entered training, but they did guide how many trees the forest keeps — so treat this as a classroom check, not an independent benchmark.
        </p>
      </div>

      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 mb-7">
        <h2 className="text-lg font-bold text-slate-900 mb-2">Held-out check</h2>
        <p className="text-sm text-slate-600 leading-relaxed">
          The {session!.split.held_out.length} held-out teaching patients ({session!.split.held_out.join(', ')}) were never used for training or for model selection.
          The same fitted model scores <strong className="text-slate-900">{(session!.held_out_evaluation.metrics.accuracy * 100).toFixed(1)}%</strong> accuracy on them
          ({session!.held_out_evaluation.predictions.filter(row => row.correct).length}/{session!.held_out_evaluation.sample_count} correct).
          With only {session!.held_out_evaluation.sample_count} synthetic patients, a few answers move this number a long way.
        </p>
      </div>
      <div className="text-center"><button onClick={onNext} className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-9 py-3.5 rounded-xl font-bold shadow-md transition-colors">Next <span aria-hidden="true">→</span></button></div>
    </div>
  )
}

// ── Screen 6 — Select New Test Patient ────────────────────────
function SelectTestPatientScreen({ patients, mode, testId, onSelect, onNext, onBack }: { patients: Patient[]; mode: Mode; testId: number | null; onSelect: (id: number) => void; onNext: () => void; onBack: () => void }) {
  const [viewingPatient, setViewingPatient] = useState<Patient | null>(null)
  return (
    <div className="max-w-7xl mx-auto px-6 py-6 animate-fade-in">
      <ProgressSteps current={5} />
      <div className="mb-4"><BackButton onClick={onBack} /></div>
      <div className="text-center mb-6">
        <h1 className="text-3xl font-bold text-slate-900 mb-2">Choose a Held-out Patient for AI Prediction</h1>
        <p className="text-slate-500 max-w-2xl mx-auto mb-3">Choose one held-out patient to run an individual prediction with the current model.</p>
        <div className="inline-flex items-center gap-2 bg-red-50 border border-red-100 rounded-lg px-4 py-2 text-sm text-red-700">{mode === 'kidney' ? 'These are the same ten test cases shown in the recorded evaluation. This request runs fresh inference; it is not a new independent test.' : 'These patients were not used for fitting or validation-based model selection.'}</div>
      </div>
      <div className="flex items-center gap-3 mb-5 max-w-3xl mx-auto"><div className="flex-1 h-px bg-red-100" /><div className="bg-red-50 border border-red-100 rounded-full px-4 py-1.5 text-xs font-semibold text-red-700">Held-out Cases — {patients.length} Patients</div><div className="flex-1 h-px bg-red-100" /></div>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8 max-w-4xl mx-auto">
        {patients.map(patient => {
          const selected = testId === patient.id
          return (
            <div key={patient.id} role="button" tabIndex={0} onClick={() => onSelect(patient.id)} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(patient.id) } }} className={`relative bg-white rounded-xl border-2 p-3 pb-9 flex flex-col items-center gap-2 transition-all cursor-pointer focus:outline-none ${selected ? 'patient-card-selected-test' : 'border-slate-200 hover:border-red-200 hover:shadow-md hover:-translate-y-0.5'}`}>
              {selected && <div className="absolute top-1.5 right-1.5 w-5 h-5 bg-red-600 rounded-full flex items-center justify-center checkmark-anim"><span className="text-white text-xs">✓</span></div>}
              <PatientAvatar id={patient.id} size={52} />
              <div className="text-center"><div className="text-xs font-semibold text-slate-800">{patient.name}</div><div className="text-xs text-slate-400 mt-0.5 font-mono">{mode === 'diabetes' ? `Age ${patient.age}` : patient.case_id}</div></div>
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

// ── Stored patient data ─────────────────────────────────────

function PatientProfile({ patient, mode, test = false }: { patient: Patient; mode: Mode; test?: boolean }) {
  return (
    <div className={`flex items-center gap-4 p-4 rounded-xl border ${test ? 'bg-red-50 border-red-100' : 'bg-blue-50 border-blue-100'}`}>
      <PatientAvatar id={patient.id} size={64} ring={test ? '#DC2626' : '#2563EB'} />
      <div>
        <h3 style={{ fontFamily: 'Outfit, sans-serif' }} className="font-bold text-slate-900 text-lg">{patient.name}</h3>
        <div className="flex flex-wrap items-center gap-2 mt-1 text-sm text-slate-500">
          <span className="font-mono">{mode === 'diabetes' ? `Age ${patient.age}` : patient.case_id}</span>
          <span>·</span>
          <span>{patient.patient_code}</span>
          {test && <span className="bg-red-100 text-red-700 px-2 py-0.5 rounded-full text-xs font-medium">Test Patient</span>}
        </div>
      </div>
    </div>
  )
}

function ModeData({ patient, mode }: { patient: Patient; mode: Mode; analysing?: boolean }) {
  if (mode === 'kidney') {
    return (
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="px-4 py-3 bg-slate-50 border-b border-slate-200">
          <span className="text-xs font-semibold text-slate-700 uppercase tracking-wider">Kidney CT Scan</span>
        </div>
        <div className="p-4">
          <KidneyCTViewer key={patient.patient_code} patient={patient} />
          <p className="text-xs text-slate-500 mt-3">KiTS19 CT images grouped by the project40 manifest. Names are fictional teaching identities.</p>
        </div>
      </div>
    )
  }
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
      <div className="px-4 py-3 bg-slate-50 border-b border-slate-200">
        <span className="text-xs font-semibold text-slate-700 uppercase tracking-wider">Diabetes Tabular Data</span>
      </div>
      <div className="divide-y divide-slate-100">
        {Object.entries(getBloodTests(patient)).map(([label, value]) => (
          <div key={label} className="flex items-center justify-between gap-4 px-4 py-3">
            <span className="text-sm text-slate-500">{label}</span>
            <span className="text-sm font-mono font-medium text-slate-800 text-right">{value}</span>
          </div>
        ))}
      </div>
      <p className="px-4 py-3 bg-blue-50 text-xs text-blue-700 border-t border-blue-100">Original synthetic diabetes data. The reference label is the target, not an input feature.</p>
    </div>
  )
}

function PatientDataModal({ patient, mode, onClose }: { patient: Patient; mode: Mode; onClose: () => void }) {
  const dialogRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const previousFocus = document.activeElement as HTMLElement | null
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    dialogRef.current?.querySelector<HTMLButtonElement>('button')?.focus()
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
      if (event.key !== 'Tab') return
      const controls = dialogRef.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled)')
      if (!controls?.length) return
      const first = controls[0], last = controls[controls.length - 1]
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus() }
      if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      document.body.style.overflow = previousOverflow
      previousFocus?.focus()
    }
  }, [onClose])

  return createPortal(
    <div className="fixed inset-0 z-50 bg-slate-950/45 p-4 flex items-center justify-center" onMouseDown={onClose}>
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label={`Patient data for ${patient.name}`}
        className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto bg-white rounded-2xl shadow-2xl p-6 animate-scale-in"
        onMouseDown={event => event.stopPropagation()}
      >
        <button type="button" onClick={onClose} aria-label="Close patient data" className="absolute top-4 right-4 w-9 h-9 rounded-lg bg-slate-100 text-slate-500 hover:bg-slate-200 flex items-center justify-center">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M3 3l10 10M13 3L3 13" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /></svg>
        </button>
        <div className="pr-12 mb-5"><PatientProfile patient={patient} mode={mode} /></div>
        <ModeData patient={patient} mode={mode} />
      </div>
    </div>, document.body
  )
}

// Shared, stable left column for detection and prediction states.
function PatientDataPanel({ patient, mode, analysing = false }: { patient: Patient; mode: Mode; analysing?: boolean }) {
  return (
    <div className="space-y-5">
      <PatientProfile patient={patient} mode={mode} test />
      <ModeData patient={patient} mode={mode} analysing={analysing} />
    </div>
  )
}

// ── Screens 6 & 7 — One stable detection interface ────────────
type DetectionState = 'ready' | 'analysing' | 'result' | 'unavailable'

const MODE_DETECTION_STAGES: Record<Mode, string[]> = {
  diabetes: [
    'Sending this patient\'s blood test values to the backend…',
    'Loading the Random Forest trained in your session…',
    'Scoring the six numerical features…',
    'Comparing the score with the 0.5 decision threshold…',
    'Returning the prediction and local explanation…',
  ],
  kidney: [
    'Loading CT scan…',
    'Preprocessing CT image…',
    'Extracting visual features…',
    'Running the segmentation model…',
    'Calculating tumour probability…',
    'Generating prediction…',
  ],
}

function KidneySegmentationPanel({ prediction }: { prediction: KidneyPrediction }) {
  const top = prediction.tumour_slices[0]
  return (
    <div className="rounded-xl border border-blue-100 bg-white shadow-sm overflow-hidden">
      <div className="px-4 py-3 bg-blue-50 border-b border-blue-100">
        <h3 className="text-sm font-bold text-blue-900">Model segmentation</h3>
        <p className="text-xs text-blue-700 mt-0.5">
          Predicted kidney (green) and tumour (red) regions on the most affected slice.
        </p>
      </div>
      {top ? (
        <div className="p-4">
          <img
            src={overlayUrl(prediction.patient_id, top.z)}
            alt={`Predicted segmentation overlay for ${prediction.case_id} slice ${top.z}`}
            className="w-full rounded-lg border border-slate-200 bg-slate-950"
          />
          <p className="mt-3 text-xs text-slate-500 font-mono">
            {prediction.case_id} · slice z={top.z} · tumour covers {(top.tumour_pixel_fraction * 100).toFixed(2)}% of the slice
          </p>
          <div className="mt-3 grid grid-cols-3 gap-2 text-center">
            {[
              ['Slices analysed', String(prediction.slice_count)],
              ['Slices with tumour', String(prediction.tumour_slice_count)],
              ['Highest tumour probability', `${(prediction.estimated_tumour_probability * 100).toFixed(1)}%`],
            ].map(([label, value]) => (
              <div key={label} className="rounded-lg bg-slate-50 border border-slate-200 px-2 py-2">
                <div className="text-[10px] uppercase tracking-wider text-slate-400">{label}</div>
                <div className="text-sm font-bold text-slate-800">{value}</div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="p-4 text-sm text-slate-600">
          No slice in this case was predicted to contain tumour tissue, so there is no overlay to display.
        </div>
      )}
      <p className="px-4 py-3 bg-slate-50 border-t border-slate-100 text-[11px] text-slate-500 leading-relaxed">
        Trained on 20 teaching cases at {prediction.selected_epoch ? `epoch ${prediction.selected_epoch}` : 'the selected epoch'};
        measured on 256×256 CT slices. Educational only — not a medical diagnosis.
      </p>
    </div>
  )
}

function PredictionExplainability({ prediction }: { prediction: PredictionResult }) {
  if (!prediction.feature_effects.length) {
    return <div className="rounded-xl border border-blue-100 bg-blue-50 p-4 text-sm text-blue-800">
      {prediction.note ?? 'No local explanation is available for this prediction.'}
    </div>
  }
  const largest = Math.max(...prediction.feature_effects.map(effect => Math.abs(effect.probability_change)), 0.0001)
  return (
    <div className="rounded-xl border border-blue-100 bg-white shadow-sm overflow-hidden">
      <div className="px-4 py-3 bg-blue-50 border-b border-blue-100">
        <h3 className="text-sm font-bold text-blue-900">What moved this prediction?</h3>
        <p className="text-xs text-blue-700 mt-0.5">Each bar swaps one value back to the training-set reference and measures the change in the model score.</p>
      </div>
      <div className="divide-y divide-slate-100">
        {prediction.feature_effects.map(effect => {
          const width = Math.max((Math.abs(effect.probability_change) / largest) * 100, 2)
          const raises = effect.direction === 'increased_score'
          const neutral = effect.direction === 'no_change'
          return (
            <div key={effect.feature_name} className="px-4 py-3">
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-sm font-medium text-slate-800">{effect.display_name}</span>
                <span className={`text-xs font-mono font-semibold ${neutral ? 'text-slate-400' : raises ? 'text-red-600' : 'text-emerald-600'}`}>
                  {effect.probability_change >= 0 ? '+' : ''}{(effect.probability_change * 100).toFixed(1)} pts
                </span>
              </div>
              <div className="mt-1.5 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                <div className={`h-full rounded-full ${neutral ? 'bg-slate-300' : raises ? 'bg-red-400' : 'bg-emerald-400'}`} style={{ width: `${width}%` }} />
              </div>
              <div className="mt-1 text-[11px] text-slate-400 font-mono">patient {effect.patient_value} · reference {effect.reference_value}</div>
            </div>
          )
        })}
      </div>
      <p className="px-4 py-3 bg-slate-50 border-t border-slate-100 text-[11px] text-slate-500 leading-relaxed">
        One-feature-at-a-time sensitivity against the training reference. Effects are not additive and do not prove causation or clinical importance.
      </p>
    </div>
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
  const [prediction, setPrediction] = useState<AnyPrediction | null>(null)
  const [failure, setFailure] = useState('')
  const stages = MODE_DETECTION_STAGES[mode]
  const kidneyResult = isKidneyPrediction(prediction) ? prediction : null
  const diabetesResult = prediction && !isKidneyPrediction(prediction) ? prediction : null
  const positive = prediction ? prediction.predicted_label === 1 : false
  const score = diabetesResult
    ? diabetesResult.estimated_diabetes_probability
    : kidneyResult
      ? kidneyResult.estimated_tumour_probability
      : 0
  const resultLabel = mode === 'diabetes'
    ? (positive ? 'Diabetes Detected' : 'No Diabetes Detected')
    : (positive ? 'Kidney Tumour Detected' : 'No Kidney Tumour Detected')

  const handleDetect = () => {
    setDetectionState('analysing')
    setStage(0)
    setPrediction(null)
    setFailure('')
    predictPatient(mode, patient.patient_code)
      .then(result => setPrediction(result))
      .catch((error: unknown) => setFailure(
        error instanceof ModelApiError ? error.message : 'The model service returned an unexpected response.',
      ))
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
    if (prediction) {
      const timer = window.setTimeout(() => setDetectionState('result'), 450)
      return () => window.clearTimeout(timer)
    }
    if (failure) setDetectionState('unavailable')
  }, [detectionState, stage, stages.length, prediction, failure])

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
          {detectionState === 'result' && diabetesResult && (
            <div className="mt-5">
              <PredictionExplainability prediction={diabetesResult} />
            </div>
          )}
          {detectionState === 'result' && kidneyResult && (
            <div className="mt-5">
              <KidneySegmentationPanel prediction={kidneyResult} />
            </div>
          )}
        </section>

        <aside className={`min-w-0 lg:sticky lg:top-6 ${detectionState === 'result' ? 'order-1 lg:order-2' : 'order-2'}`}>
          <div className="mb-4 flex items-center justify-between">
            <h2 style={{ fontFamily: 'Outfit, sans-serif' }} className="text-xl font-bold text-slate-900">
              {detectionState === 'result' ? 'AI Prediction Result' : detectionState === 'analysing' ? 'AI Analysing' : 'AI Detection'}
            </h2>
            <span className="text-xs font-mono text-slate-400">
              {mode === 'diabetes' ? 'RANDOM FOREST' : 'U-NET'}
            </span>
          </div>
          <div className={`bg-white rounded-2xl border border-slate-200 shadow-sm p-6 flex flex-col transition-all duration-300 ${detectionState === 'result' ? 'lg:min-h-[31rem]' : 'min-h-[31rem]'}`}>
            {detectionState === 'ready' && (
              <div className="flex flex-col h-full animate-fade-in">
                <div className="bg-gradient-to-br from-blue-50 to-slate-50 rounded-2xl border border-blue-100 p-7 text-center mb-5">
                  <div className="w-20 h-20 rounded-full bg-white border border-blue-200 shadow-sm flex items-center justify-center mx-auto mb-4"><AIReadyIcon /></div>
                  <h3 className="text-lg font-bold text-slate-900 mb-2">Your {mode === 'diabetes' ? 'Random Forest' : 'segmentation'} model is ready.</h3>
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

            {detectionState === 'result' && diabetesResult && (
              <div className="flex flex-col h-full animate-scale-in">
                <div className={`rounded-2xl border-2 p-6 text-center ${positive ? 'bg-red-50 border-red-200' : 'bg-emerald-50 border-emerald-200'}`}>
                  <div className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">Predicted Outcome</div>
                  <div className={`text-2xl font-extrabold ${positive ? 'text-red-700' : 'text-emerald-700'}`}>{resultLabel}</div>
                  <div className="text-sm text-slate-500 mt-2">Decision threshold {diabetesResult.prediction_threshold ?? 0.5} · {diabetesResult.best_tree_count} trees</div>
                </div>
                <div className="my-5 rounded-xl border border-slate-200 p-5">
                  <div className="flex items-end justify-between mb-3">
                    <div>
                      <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Model-Estimated Diabetes Score</div>
                      <div className="text-xs text-slate-400 mt-1">
                        {diabetesResult.source === 'session_model' ? 'From the model trained in this session' : 'From the committed demonstration model'}
                      </div>
                    </div>
                    <div className="text-3xl font-bold text-blue-600">{(score * 100).toFixed(1)}%</div>
                  </div>
                  <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden"><div className="h-full bg-blue-600 rounded-full confidence-bar" style={{ width: `${Math.min(score * 100, 100)}%` }} /></div>
                  <div className="mt-3 text-xs text-slate-500">
                    Reference label in the dataset: <strong className={diabetesResult.actual_label === 1 ? 'text-red-600' : 'text-emerald-600'}>
                      {diabetesResult.actual_label === 1 ? 'Diabetes' : 'No Diabetes'}
                    </strong> — the model {diabetesResult.actual_label === diabetesResult.predicted_label ? 'matched' : 'did not match'} it.
                  </div>
                </div>
                <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-xs text-amber-800 leading-relaxed mb-5">
                  <strong>Educational prototype:</strong> this score was produced by the Random Forest fitted on the server from this session's 20 training patients. It is a model-estimated educational score, not a calibrated clinical risk or a medical diagnosis.
                </div>
                <div className="mt-auto hidden lg:flex gap-3">
                  <button onClick={onTestAnother} className="flex-1 border-2 border-red-200 text-red-600 hover:bg-red-50 px-3 py-3 rounded-xl font-semibold text-sm transition-colors">Test Another</button>
                  <button onClick={onRestart} className="flex-1 bg-slate-800 hover:bg-slate-900 text-white px-3 py-3 rounded-xl font-semibold text-sm transition-colors">Start Over</button>
                </div>
              </div>
            )}

            {detectionState === 'result' && kidneyResult && (
              <div className="flex flex-col h-full animate-scale-in">
                <div className={`rounded-2xl border-2 p-6 text-center ${positive ? 'bg-red-50 border-red-200' : 'bg-emerald-50 border-emerald-200'}`}>
                  <div className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">Predicted Outcome</div>
                  <div className={`text-2xl font-extrabold ${positive ? 'text-red-700' : 'text-emerald-700'}`}>{resultLabel}</div>
                  <div className="text-sm text-slate-500 mt-2">Epoch {kidneyResult.selected_epoch} · {kidneyResult.slice_count} CT slices analysed</div>
                </div>
                <div className="my-5 rounded-xl border border-slate-200 p-5">
                  <div className="flex items-end justify-between mb-3">
                    <div>
                      <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Highest Tumour Probability</div>
                      <div className="text-xs text-slate-400 mt-1">Largest softmax score across every pixel and slice</div>
                    </div>
                    <div className="text-3xl font-bold text-blue-600">{(score * 100).toFixed(1)}%</div>
                  </div>
                  <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden"><div className="h-full bg-blue-600 rounded-full confidence-bar" style={{ width: `${Math.min(score * 100, 100)}%` }} /></div>
                  <div className="mt-3 text-xs text-slate-500">
                    The model marked <strong className="text-slate-800">{kidneyResult.tumour_slice_count}</strong> of {kidneyResult.slice_count} slices as containing tumour
                    ({(kidneyResult.tumour_slice_ratio * 100).toFixed(0)}%).
                    {kidneyResult.reference_tumour_slice_count !== undefined && (
                      <> The reference mask marks {kidneyResult.reference_tumour_slice_count} slices — this case {kidneyResult.actual_label === 1 ? 'does' : 'does not'} contain tumour.</>
                    )}
                  </div>
                </div>
                <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-xs text-amber-800 leading-relaxed mb-5">
                  <strong>Educational prototype:</strong> this segmentation came from the local U-Net checkpoint trained on 20 teaching cases. It is not a medical diagnosis.
                </div>
                <div className="mt-auto hidden lg:flex gap-3">
                  <button onClick={onTestAnother} className="flex-1 border-2 border-red-200 text-red-600 hover:bg-red-50 px-3 py-3 rounded-xl font-semibold text-sm transition-colors">Test Another</button>
                  <button onClick={onRestart} className="flex-1 bg-slate-800 hover:bg-slate-900 text-white px-3 py-3 rounded-xl font-semibold text-sm transition-colors">Start Over</button>
                </div>
              </div>
            )}

            {detectionState === 'unavailable' && (
              <div className="flex flex-col h-full animate-fade-in">
                <div className="rounded-2xl border-2 border-amber-200 bg-amber-50 p-6">
                  <div className="text-xs font-semibold uppercase tracking-wider text-amber-700 mb-2">Prediction unavailable</div>
                  <h3 className="text-lg font-bold text-amber-900 mb-2">The {mode === 'kidney' ? 'kidney segmentation' : 'diabetes'} model did not run.</h3>
                  <p className="text-sm text-amber-800 leading-relaxed" role="alert">{failure}</p>
                  <p className="text-xs text-amber-700 mt-3">No prediction is shown because the backend could not produce one. The stored patient data on the left is unchanged.</p>
                </div>
                <div className="mt-auto pt-5 flex gap-3">
                  <button onClick={handleDetect} className="flex-1 border-2 border-amber-300 text-amber-700 hover:bg-amber-50 px-3 py-3 rounded-xl font-semibold text-sm transition-colors">Try again</button>
                  <button onClick={onBack} className="flex-1 bg-slate-800 hover:bg-slate-900 text-white px-3 py-3 rounded-xl font-semibold text-sm transition-colors">Back</button>
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
const WORKFLOW_STORAGE_KEY = 'healthcare-education-workflow-json-v2'

function readSavedWorkflow() {
  try {
    const saved = JSON.parse(window.sessionStorage.getItem(WORKFLOW_STORAGE_KEY) ?? '{}')
    if (saved.mode !== routeFromPath(window.location.pathname)?.mode) throw new Error('Different module')
    const validIds = new Set(Array.from({ length: 40 }, (_, index) => index + 1))
    const trainingIds = Array.isArray(saved.trainingIds)
      ? saved.trainingIds.filter((id: unknown) => typeof id === 'number' && validIds.has(id)).slice(0, 20)
      : []
    const testId = validIds.has(saved.testId) ? saved.testId as number : null
    return {
      trainingIds: new Set<number>(trainingIds),
      testId,
      trainingComplete: saved.trainingComplete === true,
      evaluationComplete: saved.evaluationComplete === true,
      trainingLogCount: Number.isInteger(saved.trainingLogCount) && saved.trainingLogCount >= 0
        ? saved.trainingLogCount as number : 0,
    }
  } catch {
    return { trainingIds: new Set<number>(), testId: null, trainingComplete: false,
      evaluationComplete: false, trainingLogCount: 0 }
  }
}

export default function App() {
  const [route, setRoute] = useState<AppRoute>(() => {
    const initialRoute = routeFromPath(window.location.pathname)
    if (!initialRoute) window.history.replaceState(null, '', '/')
    return initialRoute ?? HOME_ROUTE
  })
  const [savedWorkflow] = useState(readSavedWorkflow)
  const [storedTrainingIds, setTrainingIds] = useState(savedWorkflow.trainingIds)
  const [testId, setTestId] = useState(savedWorkflow.testId)
  const [trainingComplete, setTrainingComplete] = useState(savedWorkflow.trainingComplete)
  const [evaluationComplete, setEvaluationComplete] = useState(savedWorkflow.evaluationComplete)
  const [trainingLogCount, setTrainingLogCount] = useState(savedWorkflow.trainingLogCount)
  const [session, setSession] = useState<ModelSession | null>(null)
  const [trainingError, setTrainingError] = useState('')
  const [trainingUnavailable, setTrainingUnavailable] = useState(false)
  const [trainingLoading, setTrainingLoading] = useState(false)
  const [trainingAttempt, setTrainingAttempt] = useState(0)
  const { screen, mode } = route
  const { patients, error, retry } = usePatients(mode)
  const eligiblePatients = patients.filter(p => mode === 'kidney' ? p.pool === 'train' : p.pool === 'development')
  const trainingIds = new Set([...storedTrainingIds].filter(id => eligiblePatients.some(p => p.id === id)))
  const trainingKey = [...trainingIds].sort((a, b) => a - b).join(',')
  const newPatients = patients.filter(p => p.pool === 'test')
  const previousMode = useRef(mode)
  useEffect(() => {
    if (previousMode.current !== mode) {
      setTrainingIds(new Set())
      setTestId(null)
      setTrainingComplete(false)
      setEvaluationComplete(false)
      setTrainingLogCount(0)
      setSession(null)
      setTrainingError('')
      setTrainingUnavailable(false)
      previousMode.current = mode
    }
  }, [mode])

  // Train on the server whenever the workflow reaches the training screen.
  useEffect(() => {
    if (screen !== 3 || !mode) return
    const ids = trainingKey ? trainingKey.split(',').map(Number) : []
    if (ids.length !== 20) return
    const codes = ids.map(id => `P${String(id).padStart(3, '0')}`)
    const signature = codes.join(',')
    if (session && session.selection.training_patient_ids.join(',') === signature) return

    let cancelled = false
    setTrainingLoading(true)
    setTrainingError('')
    setTrainingUnavailable(false)
    setTrainingComplete(false)
    setTrainingLogCount(0)
    trainModel(mode, codes)
      .then(result => {
        if (cancelled) return
        setSession(result)
      })
      .catch((reason: unknown) => {
        if (cancelled) return
        setSession(null)
        setTrainingUnavailable(reason instanceof ModelApiError && reason.unavailable)
        setTrainingError(reason instanceof Error ? reason.message : 'The model service could not train this selection.')
      })
      .finally(() => {
        if (!cancelled) setTrainingLoading(false)
      })
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [screen, mode, trainingKey, trainingAttempt])

  useEffect(() => {
    const onPopState = () => {
      const nextRoute = routeFromPath(window.location.pathname)
      if (!nextRoute) window.history.replaceState(null, '', '/')
      setRoute(nextRoute ?? HOME_ROUTE)
    }
    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  }, [])

  useEffect(() => {
    try {
      window.sessionStorage.setItem(WORKFLOW_STORAGE_KEY, JSON.stringify({
        mode, trainingIds: [...storedTrainingIds], testId, trainingComplete, evaluationComplete, trainingLogCount,
      }))
    } catch {
      // The routes still work when browser storage is unavailable.
    }
  }, [mode, storedTrainingIds, testId, trainingComplete, evaluationComplete, trainingLogCount])

  const navigateTo = (nextScreen: ScreenId, nextMode: Mode | null = mode, patientId: number | null = null) => {
    const nextRoute: AppRoute = { screen: nextScreen, mode: nextScreen === 1 ? null : nextMode, testId: patientId }
    const nextPath = pathForRoute(nextRoute)
    if (window.location.pathname !== nextPath) window.history.pushState(null, '', nextPath)
    setRoute(nextRoute)
    window.scrollTo(0, 0)
  }

  // Keep the workflow consistent after a reload: the fitted model lives on the server.
  useEffect(() => {
    if (!mode) return
    if (screen === 3 && trainingIds.size !== 20) {
      navigateTo(2)
      return
    }
    if (screen >= 4 && !session && !trainingUnavailable) {
      navigateTo(trainingIds.size === 20 ? 3 : 2)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [screen, mode, session, trainingIds.size])

  const resetWorkflow = () => {
    navigateTo(1)
    setTrainingIds(new Set())
    setTestId(null)
    setTrainingComplete(false)
    setEvaluationComplete(false)
    setTrainingLogCount(0)
    setSession(null)
    setTrainingError('')
    setTrainingUnavailable(false)
  }

  const handleSelectMode = (m: Mode) => {
    if (mode === m) {
      navigateTo(2, m)
      return
    }
    setTestId(null)
    setTrainingComplete(false)
    setEvaluationComplete(false)
    setTrainingLogCount(0)
    setSession(null)
    setTrainingError('')
    setTrainingUnavailable(false)
    navigateTo(2, m)
  }

  const handleToggleTraining = (id: number) => {
    if (!eligiblePatients.some(p => p.id === id)) return
    setTrainingIds(() => {
      const next = new Set(trainingIds)
      if (next.has(id)) next.delete(id)
      else if (next.size < 20) next.add(id)
      return next
    })
    setTrainingComplete(false)
    setEvaluationComplete(false)
    setTrainingLogCount(0)
    setTestId(null)
    setSession(null)
    setTrainingError('')
  }

  const handleRandomTraining = () => {
    const shuffledIds = eligiblePatients.map(patient => patient.id)
    for (let index = shuffledIds.length - 1; index > 0; index--) {
      const swapIndex = Math.floor(Math.random() * (index + 1))
      ;[shuffledIds[index], shuffledIds[swapIndex]] = [shuffledIds[swapIndex], shuffledIds[index]]
    }
    const repeatsCurrentSelection =
      trainingIds.size === 20 && shuffledIds.slice(0, 20).every(id => trainingIds.has(id))
    if (repeatsCurrentSelection && shuffledIds.length > 20) {
      ;[shuffledIds[0], shuffledIds[20]] = [shuffledIds[20], shuffledIds[0]]
    }
    setTrainingIds(new Set(shuffledIds.slice(0, 20)))
    setTrainingComplete(false)
    setEvaluationComplete(false)
    setTrainingLogCount(0)
    setTestId(null)
    setSession(null)
    setTrainingError('')
  }

  const handleSelectTest = (id: number) => {
    setTestId(id)
  }

  const evaluationPatients = session
    ? patients.filter(patient => evaluationPatientCodes(session).includes(patient.patient_code))
    : []
  const testPatient = route.testId ? newPatients.find(patient => patient.id === route.testId) ?? null : null

  if (mode && !patients.length) return (
    <div className="min-h-screen bg-slate-50"><Header onHome={() => navigateTo(1)} />
      <div className="max-w-xl mx-auto my-24 p-8 rounded-2xl border border-slate-200 bg-white text-center">
        {error ? <><p role="alert" className="text-red-700 mb-4">{error} Start the patient API and try again.</p><button className="bg-blue-600 text-white rounded-lg px-5 py-2" onClick={retry}>Retry</button></> : <p role="status">Loading patient data…</p>}
      </div>
    </div>
  )

  return (
    <div className="min-h-screen" style={{ background: '#F8FAFC' }}>
      <Header onHome={() => navigateTo(1)} />
      {screen >= 3 && (
        <p className="text-center text-sm p-3 border-b bg-emerald-50 border-emerald-200 text-emerald-900">
          {mode === 'diabetes'
            ? 'Training and predictions run the real Random Forest in the local Python backend. All patients are fictional and results are for education only.'
            : 'This activity replays recorded U-Net training and evaluation. Patient predictions run the included checkpoint on the backend. Educational only.'}
        </p>
      )}
      {screen === 7 && !testPatient && <div role="alert" className="p-12 text-center">This patient is not in the held-out test pool. <button className="text-blue-600 underline" onClick={() => navigateTo(6)}>Choose a test patient</button></div>}
      {screen === 1 && <Screen1 onSelect={handleSelectMode} />}
      {screen === 2 && (
        <Screen2
          patients={patients}
          trainingIds={trainingIds}
          mode={mode!}
          onToggle={handleToggleTraining}
          onRandomSelect={handleRandomTraining}
          onNext={() => navigateTo(3)}
          onBack={() => navigateTo(1)}
        />
      )}
      {screen === 3 && mode && (
        <Screen3
          mode={mode}
          session={session}
          loading={trainingLoading}
          error={trainingError}
          unavailable={trainingUnavailable}
          selectedCount={trainingIds.size}
          completed={trainingComplete}
          initialVisibleCount={trainingLogCount}
          onProgress={setTrainingLogCount}
          onTrainingComplete={() => setTrainingComplete(true)}
          onContinue={() => navigateTo(4)}
          onBack={() => navigateTo(2)}
          onRetry={() => setTrainingAttempt(value => value + 1)}
        />
      )}
      {screen === 4 && mode && (
        <EvaluationDataScreen
          patients={evaluationPatients}
          mode={mode}
          completed={evaluationComplete}
          onComplete={() => { setEvaluationComplete(true); navigateTo(5) }}
          onBack={() => navigateTo(3)}
        />
      )}
      {screen === 5 && mode && (
        <EvaluationResultsScreen
          patients={patients}
          mode={mode}
          session={session}
          onNext={() => navigateTo(6)}
          onBack={() => navigateTo(4)}
        />
      )}
      {screen === 6 && mode && (
        <SelectTestPatientScreen
          patients={newPatients}
          mode={mode}
          testId={testId}
          onSelect={handleSelectTest}
          onNext={() => navigateTo(7, mode, testId)}
          onBack={() => navigateTo(5)}
        />
      )}
      {screen === 7 && testPatient && mode && (
        <Screen5
          patient={testPatient}
          mode={mode}
          onTestAnother={() => { setTestId(null); navigateTo(6) }}
          onRestart={resetWorkflow}
          onBack={() => navigateTo(6)}
        />
      )}
    </div>
  )
}
