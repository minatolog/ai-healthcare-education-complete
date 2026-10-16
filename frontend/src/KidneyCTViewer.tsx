import { useEffect, useId, useState } from 'react'
import type { Patient } from './patients'

type Slice = { z: number; url: string }

export default function KidneyCTViewer({ patient }: { patient: Patient }) {
  const sliderId = useId()
  const [series, setSeries] = useState<{ code: string; slices: Slice[] } | null>(null)
  const [index, setIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [error, setError] = useState('')
  const [loadedUrl, setLoadedUrl] = useState('')
  const [failedUrl, setFailedUrl] = useState('')
  const [attempt, setAttempt] = useState(0)
  const slices = series?.code === patient.patient_code ? series.slices : []
  const slice = slices[index]

  useEffect(() => {
    const controller = new AbortController()
    setSeries(null)
    setPlaying(false)
    setError('')
    setLoadedUrl('')
    setFailedUrl('')
    fetch(`/api/kidney/patients/${patient.patient_code}/ct`, { signal: controller.signal })
      .then(async response => {
        if (!response.ok) throw new Error('CT series could not be loaded.')
        const data = await response.json()
        if (!Array.isArray(data.slices) || data.slices.length !== patient.slice_count || !data.slices.length) {
          throw new Error('CT series is incomplete.')
        }
        if (!controller.signal.aborted) {
          setSeries({ code: patient.patient_code, slices: data.slices })
          setIndex(Math.floor(data.slices.length / 2))
        }
      })
      .catch(err => {
        if (!controller.signal.aborted) setError(err instanceof Error ? err.message : 'CT series could not be loaded.')
      })
    return () => controller.abort()
  }, [patient.patient_code, patient.slice_count, attempt])

  useEffect(() => {
    if (!playing || !slices.length) return
    const timer = window.setInterval(() => setIndex(current => (current + 1) % slices.length), 250)
    return () => window.clearInterval(timer)
  }, [playing, slices.length])

  useEffect(() => {
    // Warm the adjacent frames only; never download all 3,757 images up front.
    for (const next of [slices[index - 1], slices[index + 1]]) {
      if (next) { const image = new Image(); image.src = next.url }
    }
  }, [slices, index])

  const move = (next: number) => { setPlaying(false); setIndex(next) }
  const buttonClass = 'rounded-lg border border-slate-200 px-3 py-2 text-xs font-semibold hover:bg-slate-100 disabled:opacity-40 disabled:cursor-not-allowed'
  if (error) return <div role="alert" className="p-5 text-sm text-red-700">{error} <button className={buttonClass} onClick={() => setAttempt(value => value + 1)}>Retry CT</button></div>
  if (!slice) return <div role="status" className="rounded-xl bg-slate-950 p-16 text-center text-slate-300">Loading CT series…</div>

  return (
    <div className="space-y-4">
      <div className="relative rounded-xl overflow-hidden bg-black aspect-square max-h-[48vh] mx-auto">
        <img key={`${slice.url}-${attempt}`} src={slice.url} alt={`CT ${patient.case_id}, source slice ${slice.z}`}
          className={`w-full h-full object-contain ${loadedUrl === slice.url ? '' : 'opacity-0'}`}
          onLoad={() => { setLoadedUrl(slice.url); setFailedUrl('') }}
          onError={() => { setFailedUrl(slice.url); setPlaying(false) }} />
        {loadedUrl !== slice.url && <div role="status" className="absolute inset-0 flex items-center justify-center text-sm text-slate-300">
          {failedUrl === slice.url ? <span role="alert">Image unavailable. <button className="underline" onClick={() => setAttempt(value => value + 1)}>Retry CT</button></span> : 'Loading slice…'}
        </div>}
        <div className="absolute top-3 left-3 rounded bg-black/70 px-2 py-1 text-[11px] text-slate-200 font-mono">CT AXIAL · {patient.case_id}</div>
        <div className="absolute bottom-3 left-3 rounded bg-black/70 px-2 py-1 text-xs text-slate-200 font-mono">Slice {index + 1} / {slices.length} · Source z = {slice.z}</div>
      </div>
      <div>
        <div className="flex justify-between text-xs text-slate-500 mb-2"><label htmlFor={sliderId}>Drag to explore CT slices</label><span>{index + 1} / {slices.length}</span></div>
        <input id={sliderId} type="range" min={0} max={slices.length - 1} step={1} value={index}
          aria-valuetext={`Slice ${index + 1} of ${slices.length}, source z ${slice.z}`}
          onChange={event => move(Number(event.target.value))}
          className="w-full h-6 cursor-ew-resize accent-blue-600" />
      </div>
      <div className="flex justify-between items-center gap-2">
        <button className={buttonClass} disabled={index === 0} onClick={() => move(index - 1)}>← Previous slice</button>
        <button className={`${buttonClass} text-blue-700`} aria-pressed={playing} onClick={() => setPlaying(value => !value)}>{playing ? 'Pause' : 'Play slices'}</button>
        <button className={buttonClass} disabled={index === slices.length - 1} onClick={() => move(index + 1)}>Next slice →</button>
      </div>
      <p className="text-xs text-slate-500">Selected CT slices in source order. Gaps in source z are expected. Use the slider or arrow keys to explore.</p>
    </div>
  )
}
