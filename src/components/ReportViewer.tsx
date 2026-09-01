import type { GradeDetails, Patient } from '../types'
import { StatusBadge } from './StatusBadge'

export function ReportViewer({ patient, detail, online, experimentalConfidence }: { patient: Patient; detail: GradeDetails; online: boolean; experimentalConfidence?: number }) {
  const tone = detail.priority === 'Priority' ? 'rose' : detail.priority === 'Review' ? 'amber' : 'teal'
  return (
    <article className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
      <header className="flex flex-wrap items-start justify-between gap-4 border-b border-slate-200 bg-slate-50 p-5">
        <div><p className="text-lg font-black tracking-tight text-slate-950">DrishtiMitra <span className="font-semibold text-teal-700">Retina-XAI</span></p><p className="mt-1 text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Screening support report · {experimentalConfidence === undefined ? 'Prototype demonstration' : 'Experimental local-model output'}</p></div>
        <div className="text-right text-xs text-slate-500"><p>Report ID: <strong className="text-slate-800">DMR-2026-0901-031</strong></p><p className="mt-1">Status: <strong className="text-slate-800">{online ? 'Sync-ready' : 'Stored locally — pending sync'}</strong></p></div>
      </header>
      <div className="grid gap-5 p-5 md:grid-cols-[1.05fr_.95fr]">
        <section><p className="label">Patient</p><dl className="mt-3 grid grid-cols-2 gap-x-5 gap-y-3 text-sm"><div><dt>Patient ID</dt><dd>{patient.id}</dd></div><div><dt>Screening date</dt><dd>{patient.screenDate}</dd></div><div><dt>Name</dt><dd>{patient.name}</dd></div><div><dt>Diabetes duration</dt><dd>{patient.diabetesDuration}</dd></div></dl></section>
        <section className="rounded-xl bg-slate-50 p-4"><p className="label">AI screening & grading</p><div className="mt-3 flex items-center justify-between"><div><p className="text-2xl font-black text-slate-950">{detail.label}</p><p className="text-sm text-slate-600">{detail.clinicalLabel}</p></div><StatusBadge tone={tone}>{detail.priority}</StatusBadge></div></section>
      </div>
      <div className="grid gap-4 border-t border-slate-100 p-5 md:grid-cols-2"><section className="rounded-xl border border-slate-200 p-4"><p className="label">Evidence included</p>{experimentalConfidence === undefined ? <ul className="mt-3 space-y-2 text-sm text-slate-700"><li>✓ Original supplied demonstration fundus image</li><li>✓ Grad-CAM++ prototype visualization</li><li>✓ Attention-gated U-Net prototype visualization</li></ul> : <ul className="mt-3 space-y-2 text-sm text-slate-700"><li>✓ Original uploaded fundus image</li><li>✓ Experimental classifier output ({(experimentalConfidence * 100).toFixed(1)}% top-class confidence)</li><li>— No patient-specific heatmap or lesion mask was generated</li></ul>}</section><section className="rounded-xl border border-slate-200 p-4"><p className="label">Recommendation</p><p className="mt-3 text-sm font-semibold leading-6 text-slate-800">{detail.recommendation}</p></section></div>
      <footer className="border-t border-slate-200 bg-amber-50 px-5 py-4 text-xs leading-5 text-amber-950"><strong>Safety principle:</strong> AI-assisted screening — not autonomous diagnosis. Image quality, screening evidence, and referral information must be interpreted by a qualified clinician; final clinical interpretation and sign-off remain with an ophthalmologist.</footer>
    </article>
  )
}
