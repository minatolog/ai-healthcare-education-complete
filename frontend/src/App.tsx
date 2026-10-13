import { useState } from 'react'

// ============================================================
// Types
// ============================================================

type Mode = 'diabetes' | 'kidney'
type Screen = 'home' | 'training-data'

type Patient = {
  id: number
  name: string
  age: number
  hasDisease: boolean
}

// ============================================================
// Workflow
// ============================================================

const STEPS = [
  'Select Training Data',
  'Train AI',
  'Evaluation Data',
  'Evaluation Results',
  'Select Test Patient',
  'AI Detection',
  'Prediction Result',
]

// ============================================================
// Patient Data
// ============================================================

const DIABETES_PATIENTS: Patient[] = [
  { id: 1, name: 'Emily Carter', age: 42, hasDisease: false },
  { id: 2, name: 'James Okonkwo', age: 58, hasDisease: true },
  { id: 3, name: 'Sarah Chen', age: 35, hasDisease: true },
  { id: 4, name: 'Michael Torres', age: 67, hasDisease: false },
  { id: 5, name: 'Priya Patel', age: 51, hasDisease: true },

  { id: 6, name: 'David Kim', age: 44, hasDisease: true },
  { id: 7, name: 'Fatima Hassan', age: 39, hasDisease: false },
  { id: 8, name: 'Robert Johnson', age: 72, hasDisease: true },
  { id: 9, name: 'Mei Lin', age: 46, hasDisease: true },
  { id: 10, name: 'Carlos Rodriguez', age: 53, hasDisease: false },

  { id: 11, name: 'Amara Diallo', age: 29, hasDisease: true },
  { id: 12, name: 'Thomas Hughes', age: 61, hasDisease: true },
  { id: 13, name: 'Yuki Tanaka', age: 38, hasDisease: false },
  { id: 14, name: 'Lakshmi Nair', age: 55, hasDisease: true },
  { id: 15, name: 'Marcus Williams', age: 48, hasDisease: true },

  { id: 16, name: 'Sophie Martin', age: 33, hasDisease: false },
  { id: 17, name: 'Hassan Al-Rashid', age: 64, hasDisease: true },
  { id: 18, name: 'Emma Johansson', age: 41, hasDisease: true },
  { id: 19, name: 'Benjamin Osei', age: 57, hasDisease: false },
  { id: 20, name: 'Natasha Petrova', age: 45, hasDisease: true },

  { id: 21, name: 'Diego Fernandez', age: 50, hasDisease: true },
  { id: 22, name: 'Aisha Mohammed', age: 36, hasDisease: false },
  { id: 23, name: "Patrick O'Brien", age: 69, hasDisease: true },
  { id: 24, name: 'Lin Wei', age: 43, hasDisease: true },
  { id: 25, name: 'Ingrid Andersen', age: 52, hasDisease: false },

  { id: 26, name: 'Kwame Mensah', age: 47, hasDisease: true },
  { id: 27, name: 'Isabella Costa', age: 31, hasDisease: true },
  { id: 28, name: 'Viktor Novak', age: 66, hasDisease: false },
  { id: 29, name: 'Zara Malik', age: 40, hasDisease: true },
  { id: 30, name: 'Samuel Adeyemi', age: 59, hasDisease: true },
]

// ============================================================
// Header
// ============================================================

function Header({ onHome }: { onHome: () => void }) {
  return (
    <header className="bg-white/90 backdrop-blur-md border-b border-slate-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <button
          onClick={onHome}
          className="flex items-center gap-2.5 group"
        >
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-blue-700 flex items-center justify-center shadow-sm">
            <svg
              width="18"
              height="18"
              viewBox="0 0 18 18"
              fill="none"
            >
              <circle cx="9" cy="9" r="3.5" fill="white" />

              <circle
                cx="9"
                cy="3"
                r="1.5"
                fill="white"
                opacity="0.7"
              />

              <circle
                cx="9"
                cy="15"
                r="1.5"
                fill="white"
                opacity="0.7"
              />

              <circle
                cx="3"
                cy="9"
                r="1.5"
                fill="white"
                opacity="0.7"
              />

              <circle
                cx="15"
                cy="9"
                r="1.5"
                fill="white"
                opacity="0.7"
              />
            </svg>
          </div>

          <span
            className="font-bold text-slate-800 group-hover:text-blue-600 transition-colors"
            style={{
              fontFamily: 'Outfit, sans-serif',
              fontSize: '17px',
            }}
          >
            Interactive AI for Healthcare
          </span>
        </button>

        <nav className="flex items-center gap-6">
          <button
            onClick={onHome}
            className="text-sm font-medium text-slate-600 hover:text-blue-600 transition-colors"
          >
            Home
          </button>

          <button className="text-sm font-medium text-slate-600 hover:text-blue-600 transition-colors">
            About
          </button>

          <button className="text-sm font-medium text-slate-600 hover:text-blue-600 transition-colors">
            How It Works
          </button>
        </nav>
      </div>
    </header>
  )
}

// ============================================================
// Screen 1 — Homepage
// ============================================================

function HomeScreen({
  onSelect,
}: {
  onSelect: (mode: Mode) => void
}) {
  return (
    <main className="max-w-7xl mx-auto px-6 py-12">

      {/* Hero */}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center mb-16 animate-fade-in">
        <div>
          <div className="inline-flex items-center gap-2 bg-blue-50 text-blue-700 px-3 py-1.5 rounded-full text-sm font-medium mb-5 border border-blue-100">
            University AI Education Platform
          </div>

          <h1
            className="text-slate-900 mb-5"
            style={{
              fontFamily: 'Outfit, sans-serif',
              fontSize: 'clamp(2rem, 4vw, 3.2rem)',
              lineHeight: 1.15,
              fontWeight: 800,
            }}
          >
            Explore How AI
            <br />

            <span className="text-blue-600">
              Learns from Healthcare
            </span>

            <br />
            Data
          </h1>

          <p className="text-slate-500 text-lg leading-relaxed mb-8 max-w-lg">
            Build and test an AI model using patient data and
            discover how artificial intelligence can support
            healthcare analysis.
          </p>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-sm text-slate-500">
              <div className="w-2 h-2 rounded-full bg-blue-500" />
              Training data
            </div>

            <div className="flex items-center gap-2 text-sm text-slate-500">
              <div className="w-2 h-2 rounded-full bg-orange-500" />
              Evaluation data
            </div>

            <div className="flex items-center gap-2 text-sm text-slate-500">
              <div className="w-2 h-2 rounded-full bg-emerald-500" />
              AI predictions
            </div>
          </div>
        </div>

        <div className="flex items-center justify-center">
          <HeroIllustration />
        </div>
      </div>

      {/* Choose AI Module */}

      <div className="text-center mb-8">
        <h2
          className="text-2xl font-bold text-slate-800 mb-2"
          style={{ fontFamily: 'Outfit, sans-serif' }}
        >
          Choose Your AI Module
        </h2>

        <p className="text-slate-500">
          Select a healthcare scenario to begin your machine-learning
          journey.
        </p>
      </div>

      {/* Disease cards */}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto">

        {/* Diabetes */}

        <button
          onClick={() => onSelect('diabetes')}
          className="group text-left bg-white rounded-2xl border border-slate-200 p-8 shadow-sm card-hover cursor-pointer focus:outline-none focus:ring-4 focus:ring-blue-100"
        >
          <div className="w-16 h-16 rounded-2xl bg-blue-50 flex items-center justify-center mb-5 group-hover:bg-blue-100 transition-colors">
            <DiabetesIcon />
          </div>

          <h3
            className="text-xl font-bold text-slate-900 mb-3 group-hover:text-blue-600 transition-colors"
            style={{ fontFamily: 'Outfit, sans-serif' }}
          >
            Diabetes AI Detection
          </h3>

          <p className="text-slate-500 text-sm leading-relaxed mb-5">
            Learn how blood-test values become numerical features
            for a linear model that predicts diabetes.
          </p>

          <div className="flex items-center text-blue-600 text-sm font-semibold gap-2">
            Start Learning
            <span>→</span>
          </div>
        </button>

        {/* Kidney */}

        <button
          onClick={() => onSelect('kidney')}
          className="group text-left bg-white rounded-2xl border border-slate-200 p-8 shadow-sm card-hover cursor-pointer focus:outline-none focus:ring-4 focus:ring-red-100"
        >
          <div className="w-16 h-16 rounded-2xl bg-red-50 flex items-center justify-center mb-5 group-hover:bg-red-100 transition-colors">
            <KidneyIcon />
          </div>

          <h3
            className="text-xl font-bold text-slate-900 mb-3 group-hover:text-red-600 transition-colors"
            style={{ fontFamily: 'Outfit, sans-serif' }}
          >
            Kidney Tumour Detection
          </h3>

          <p className="text-slate-500 text-sm leading-relaxed mb-5">
            Learn how a computer vision model processes kidney CT
            scans to predict whether a tumour is present.
          </p>

          <div className="flex items-center text-red-600 text-sm font-semibold gap-2">
            Start Learning
            <span>→</span>
          </div>
        </button>
      </div>

      {/* How it works */}

      <div className="mt-16 bg-white rounded-2xl border border-slate-200 p-8 max-w-4xl mx-auto">
        <h3
          className="text-lg font-bold text-slate-800 mb-6 text-center"
          style={{ fontFamily: 'Outfit, sans-serif' }}
        >
          How It Works
        </h3>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {STEPS.map((step, index) => (
            <div
              key={step}
              className="flex flex-col items-center text-center gap-2"
            >
              <div className="w-10 h-10 rounded-xl bg-blue-50 border border-blue-100 flex items-center justify-center text-blue-600 font-bold text-sm">
                {index + 1}
              </div>

              <span className="text-xs text-slate-600 font-medium">
                {step}
              </span>
            </div>
          ))}
        </div>
      </div>
    </main>
  )
}

// ============================================================
// Workflow progress
// ============================================================

function ProgressSteps() {
  return (
    <div className="max-w-4xl mx-auto mb-8">
      <div className="flex items-start justify-between">
        {STEPS.map((step, index) => (
          <div
            key={step}
            className="flex-1 flex items-start"
          >
            <div className="flex flex-col items-center flex-1 min-w-0">
              <div
                className={`w-9 h-9 rounded-full flex items-center justify-center text-sm font-bold border-2 ${
                  index === 0
                    ? 'bg-blue-600 border-blue-600 text-white shadow-md'
                    : 'bg-white border-slate-200 text-slate-400'
                }`}
              >
                {index + 1}
              </div>

              <span
                className={`mt-2 text-[10px] text-center leading-tight ${
                  index === 0
                    ? 'text-blue-600 font-semibold'
                    : 'text-slate-400'
                }`}
              >
                {step}
              </span>
            </div>

            {index < STEPS.length - 1 && (
              <div className="h-px bg-slate-200 flex-1 mt-[18px] -mx-2" />
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

// ============================================================
// Patient Avatar
// ============================================================

function PatientAvatar({
  patient,
}: {
  patient: Patient
}) {
  const skinTones = [
    '#F2C49B',
    '#9A5E35',
    '#D49A6A',
    '#6F3D2B',
    '#E4B184',
  ]

  const hairTones = [
    '#2B2118',
    '#7C3F1D',
    '#D49A00',
    '#3A2518',
    '#B94A32',
  ]

  const skin =
    skinTones[patient.id % skinTones.length]

  const hair =
    hairTones[patient.id % hairTones.length]

  return (
    <svg
      width="44"
      height="44"
      viewBox="0 0 44 44"
      aria-hidden="true"
    >
      <circle
        cx="22"
        cy="22"
        r="21"
        fill="#F8FAFC"
      />

      <circle
        cx="22"
        cy="19"
        r="10"
        fill={skin}
      />

      <path
        d="M12 18c1-8 5-11 10-11 6 0 10 4 10 11-4-4-16-4-20 0Z"
        fill={hair}
      />

      <circle
        cx="18.5"
        cy="19"
        r="1"
        fill="#1E293B"
      />

      <circle
        cx="25.5"
        cy="19"
        r="1"
        fill="#1E293B"
      />

      <path
        d="M19 24c2 1.5 4 1.5 6 0"
        stroke="#7C4A3A"
        strokeWidth="1.2"
        strokeLinecap="round"
      />

      <path
        d="M10 42c1-9 6-13 12-13s11 4 12 13"
        fill="#DBEAFE"
      />
    </svg>
  )
}

// ============================================================
// Screen 2 — Select Training Data
// ============================================================

function TrainingDataScreen({
  mode,
  onBack,
}: {
  mode: Mode
  onBack: () => void
}) {
  const [selectedPatients, setSelectedPatients] =
    useState<number[]>([])

  const patients = DIABETES_PATIENTS

  // ----------------------------------------------------------
  // Select patient
  // ----------------------------------------------------------

  const togglePatient = (id: number) => {
    setSelectedPatients((current) => {
      if (current.includes(id)) {
        return current.filter(
          (patientId) => patientId !== id,
        )
      }

      if (current.length >= 20) {
        return current
      }

      return [...current, id]
    })
  }

  // ----------------------------------------------------------
  // Random selection
  // ----------------------------------------------------------

  const selectRandomPatients = () => {
    const ids = patients.map(
      (patient) => patient.id,
    )

    const shuffled = [...ids].sort(
      () => Math.random() - 0.5,
    )

    setSelectedPatients(
      shuffled.slice(0, 20),
    )
  }

  return (
    <main className="max-w-6xl mx-auto px-6 py-8 pb-28">

      {/* Progress */}

      <ProgressSteps />

      {/* Back */}

      <button
        onClick={onBack}
        className="mb-5 inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-600 shadow-sm hover:bg-slate-50"
      >
        ← Back
      </button>

      {/* Heading */}

      <section className="text-center mb-6">
        <p className="text-xs font-semibold uppercase tracking-wide text-blue-600 mb-2">
          {mode === 'diabetes'
            ? 'Diabetes AI Detection'
            : 'Kidney Tumour Detection'}
        </p>

        <h1
          className="text-3xl font-bold text-slate-900 mb-2"
          style={{
            fontFamily: 'Outfit, sans-serif',
          }}
        >
          Build Your AI Training Dataset
        </h1>

        <p className="text-slate-500">
          Select exactly{' '}
          <span className="font-bold text-blue-600">
            20 patients
          </span>{' '}
          to create the Training Dataset.
        </p>
      </section>

      {/* Explanation */}

      <div className="max-w-3xl mx-auto rounded-lg border border-blue-100 bg-blue-50 px-5 py-4 text-sm text-blue-700 mb-4">
        ⓘ The 20 patients you select will be used to
        train the AI model. The remaining 10 patients
        will automatically become the Evaluation Dataset
        and will be used to evaluate how well your
        trained model performs on unseen data.
      </div>

      {/* Dataset explanation */}

      <div className="flex flex-wrap justify-center items-center gap-3 mb-4 text-sm font-semibold">

        <span className="rounded-lg bg-slate-100 px-4 py-2 text-slate-600">
          30 Dataset Patients
        </span>

        <span className="text-slate-400">
          →
        </span>

        <span className="rounded-lg border border-blue-200 bg-blue-50 px-4 py-2 text-blue-600">
          20 Training Patients
        </span>

        <span className="text-slate-400">
          +
        </span>

        <span className="rounded-lg border border-orange-200 bg-orange-50 px-4 py-2 text-orange-600">
          10 Evaluation Patients
        </span>
      </div>

      {/* Random button */}

      <div className="flex justify-center mb-7">
        <button
          onClick={selectRandomPatients}
          className="rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white shadow-md hover:bg-blue-700"
        >
          ↝ Randomly Select 20 Patients
        </button>
      </div>

      {/* Patient cards */}

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
        {patients.map((patient) => {
          const selected =
            selectedPatients.includes(
              patient.id,
            )

          return (
            <button
              key={patient.id}
              onClick={() =>
                togglePatient(patient.id)
              }
              aria-pressed={selected}
              className={`relative min-h-[150px] rounded-xl border bg-white p-4 text-center shadow-sm transition-all hover:-translate-y-0.5 hover:shadow-md ${
                selected
                  ? 'border-blue-500 ring-2 ring-blue-100'
                  : 'border-slate-200'
              }`}
            >
              {/* Selected indicator */}

              {selected && (
                <span className="absolute right-2 top-2 w-5 h-5 rounded-full bg-blue-600 text-white text-xs flex items-center justify-center">
                  ✓
                </span>
              )}

              {/* Avatar */}

              <div className="flex justify-center mb-2">
                <PatientAvatar
                  patient={patient}
                />
              </div>

              {/* Patient info */}

              <p className="text-xs font-bold text-slate-800">
                {patient.name}
              </p>

              <p className="text-[11px] text-slate-400 mb-2">
                Age {patient.age}
              </p>

              {/* Ground truth label */}

              <span
                className={`inline-flex rounded-full px-2 py-1 text-[10px] font-semibold ${
                  patient.hasDisease
                    ? 'bg-red-50 text-red-500'
                    : 'bg-blue-50 text-blue-600'
                }`}
              >
                {patient.hasDisease
                  ? mode === 'diabetes'
                    ? 'Diabetes'
                    : 'Tumour'
                  : mode === 'diabetes'
                    ? 'No Diabetes'
                    : 'No Tumour'}
              </span>

              {/* Details icon */}

              <span className="absolute bottom-2 right-2 w-6 h-6 rounded-md border border-slate-200 bg-slate-50 text-slate-400 flex items-center justify-center text-xs">
                ▧
              </span>
            </button>
          )
        })}
      </div>

      {/* Bottom selection bar */}

      <div className="fixed bottom-0 left-0 right-0 border-t border-slate-200 bg-white/95 backdrop-blur-md shadow-[0_-4px_20px_rgba(15,23,42,0.06)]">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center gap-5">

          <div className="flex-1">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-semibold text-slate-700">
                Training Patients Selected
              </span>

              <span className="text-sm font-bold text-slate-700">
                {selectedPatients.length} / 20
              </span>
            </div>

            <div className="h-2 rounded-full bg-slate-100 overflow-hidden">
              <div
                className="h-full rounded-full bg-blue-600 transition-all"
                style={{
                  width: `${
                    (selectedPatients.length /
                      20) *
                    100
                  }%`,
                }}
              />
            </div>
          </div>

          <button
            disabled={
              selectedPatients.length !== 20
            }
            className="rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white disabled:bg-slate-100 disabled:text-slate-400 disabled:cursor-not-allowed"
          >
            Next →
          </button>
        </div>
      </div>
    </main>
  )
}

// ============================================================
// Hero Illustration
// ============================================================

function HeroIllustration() {
  return (
    <div className="relative w-[380px] h-[300px]">
      <svg
        width="380"
        height="300"
        viewBox="0 0 380 300"
        fill="none"
      >
        <circle
          cx="190"
          cy="150"
          r="130"
          fill="#EFF6FF"
        />

        <circle
          cx="190"
          cy="150"
          r="100"
          fill="#DBEAFE"
          opacity="0.5"
        />

        {/* Input */}

        <circle
          cx="65"
          cy="150"
          r="28"
          fill="#FEF3C7"
        />

        <circle
          cx="65"
          cy="143"
          r="10"
          fill="#F5C89A"
        />

        {/* Network */}

        {[90, 130, 170, 210].map(
          (y) => (
            <circle
              key={`left-${y}`}
              cx="170"
              cy={y}
              r="10"
              fill="#60A5FA"
            />
          ),
        )}

        {[110, 150, 190].map(
          (y) => (
            <circle
              key={`right-${y}`}
              cx="230"
              cy={y}
              r="10"
              fill="#2563EB"
            />
          ),
        )}

        {/* Output */}

        <circle
          cx="320"
          cy="150"
          r="28"
          fill="#F0FDF4"
          stroke="#86EFAC"
          strokeWidth="2"
        />

        <path
          d="M308 150l7 7 16-17"
          stroke="#16A34A"
          strokeWidth="3"
          fill="none"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        <text
          x="65"
          y="190"
          textAnchor="middle"
          fill="#94A3B8"
          fontSize="10"
        >
          Patient
        </text>

        <text
          x="200"
          y="250"
          textAnchor="middle"
          fill="#2563EB"
          fontSize="10"
          fontWeight="600"
        >
          Neural Network
        </text>

        <text
          x="320"
          y="190"
          textAnchor="middle"
          fill="#94A3B8"
          fontSize="10"
        >
          Result
        </text>
      </svg>
    </div>
  )
}

// ============================================================
// Diabetes Icon
// ============================================================

function DiabetesIcon() {
  return (
    <svg
      width="32"
      height="32"
      viewBox="0 0 32 32"
      fill="none"
    >
      <path
        d="M16 4C16 4 10 8 10 14C10 17.31 12.69 20 16 20C19.31 20 22 17.31 22 14C22 8 16 4 16 4Z"
        fill="#BFDBFE"
        stroke="#2563EB"
        strokeWidth="1.5"
      />

      <circle
        cx="24"
        cy="10"
        r="4"
        fill="#DBEAFE"
        stroke="#2563EB"
        strokeWidth="1.2"
      />

      <text
        x="24"
        y="13"
        textAnchor="middle"
        fill="#2563EB"
        fontSize="6"
        fontWeight="700"
      >
        G
      </text>
    </svg>
  )
}

// ============================================================
// Kidney Icon
// ============================================================

function KidneyIcon() {
  return (
    <svg
      width="32"
      height="32"
      viewBox="0 0 32 32"
      fill="none"
    >
      <path
        d="M10 8C6 8 4 12 4 16C4 22 8 26 12 26C14 26 16 24 16 22C16 20 14.5 19 14 17C13 14 14 10 16 10C18 10 19 14 18 17C17.5 19 16 20 16 22C16 24 18 26 20 26C24 26 28 22 28 16C28 12 26 8 22 8C20 8 18 9 16 9C14 9 12 8 10 8Z"
        fill="#FECDD3"
        stroke="#DC2626"
        strokeWidth="1.5"
      />

      <circle
        cx="22"
        cy="13"
        r="4"
        fill="#FCA5A5"
        stroke="#DC2626"
        strokeWidth="1.2"
      />
    </svg>
  )
}

// ============================================================
// Main App
// ============================================================

export default function App() {
  const [mode, setMode] =
    useState<Mode | null>(null)

  const [screen, setScreen] =
    useState<Screen>('home')

  // Return to homepage

  const goHome = () => {
    setMode(null)
    setScreen('home')
  }

  // Disease selected:
  // immediately enter Select Training Data

  const handleSelectMode = (
    selectedMode: Mode,
  ) => {
    setMode(selectedMode)
    setScreen('training-data')
  }

  // Back from training data

  const handleBackToHome = () => {
    setMode(null)
    setScreen('home')
  }

  return (
    <div
      className="min-h-screen"
      style={{
        background: '#F8FAFC',
      }}
    >
      <Header onHome={goHome} />

      {screen === 'home' && (
        <HomeScreen
          onSelect={handleSelectMode}
        />
      )}

      {screen === 'training-data' &&
        mode && (
          <TrainingDataScreen
            mode={mode}
            onBack={handleBackToHome}
          />
        )}
    </div>
  )
}