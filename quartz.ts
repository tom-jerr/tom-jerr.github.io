import { loadQuartzConfig, loadQuartzLayout } from "./quartz/plugins/loader/config-loader"
import { componentRegistry } from "./quartz/components/registry"
import { explorerOptions, recentNotesOptions } from "./quartz.navigation"

// Config loading also constructs the page dispatcher, so overrides must be
// registered before it captures the layouts used by emitted pages.
componentRegistry.setOptionOverrides("@quartz-community/explorer", explorerOptions)
componentRegistry.setOptionOverrides("@quartz-community/recent-notes", recentNotesOptions)
const config = await loadQuartzConfig()
export default config
export const layout = await loadQuartzLayout()
