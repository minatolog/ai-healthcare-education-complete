export type Mode = 'diabetes' | 'kidney'
export type ScreenId = 1 | 2 | 3 | 4 | 5 | 6 | 7

export type AppRoute = {
  screen: ScreenId
  mode: Mode | null
  testId: number | null
}

export const HOME_ROUTE: AppRoute = { screen: 1, mode: null, testId: null }

const SCREEN_PATHS: Record<2 | 3 | 4 | 5 | 6, string> = {
  2: 'patients',
  3: 'training',
  4: 'evaluation',
  5: 'results',
  6: 'prediction',
}

export function pathForRoute(route: AppRoute): string {
  if (route.screen === 1) return '/'
  if (!route.mode) throw new Error('A disease mode is required for this page')
  if (route.screen === 7) {
    if (route.testId === null || !Number.isInteger(route.testId) || route.testId < 1 || route.testId > 40) {
      throw new Error('A valid prediction patient is required')
    }
    return `/${route.mode}/prediction/${route.testId}`
  }
  return `/${route.mode}/${SCREEN_PATHS[route.screen]}`
}

export function routeFromPath(pathname: string): AppRoute | null {
  const segments = pathname.split('/').filter(Boolean)
  if (segments.length === 0) return HOME_ROUTE

  const [mode, page, patientId] = segments
  if (mode !== 'diabetes' && mode !== 'kidney') return null
  if (segments.length === 2) {
    const entry = Object.entries(SCREEN_PATHS).find(([, path]) => path === page)
    return entry ? { screen: Number(entry[0]) as ScreenId, mode, testId: null } : null
  }
  if (segments.length === 3 && page === 'prediction' && /^(?:[1-9]|[1-3]\d|40)$/.test(patientId)) {
    return { screen: 7, mode, testId: Number(patientId) }
  }
  return null
}
