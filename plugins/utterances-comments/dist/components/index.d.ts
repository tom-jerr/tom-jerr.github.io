import { QuartzComponent } from '@quartz-community/types';

interface UtterancesCommentsOptions {
    repo: `${string}/${string}`;
    issueTerm?: "pathname" | "url" | "title" | "og:title";
    label?: string;
    lightTheme?: string;
    darkTheme?: string;
}
declare const _default: (opts?: UtterancesCommentsOptions) => QuartzComponent;

export { _default as UtterancesComments, type UtterancesCommentsOptions };
