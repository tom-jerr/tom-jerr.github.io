// src/components/styles/utterances.scss
var utterances_default = ".utterances-comments {\n  width: 100%;\n  min-height: 4rem;\n  margin-top: 2rem;\n}\n\n.utterances-comments .utterances,\n.utterances-comments .utterances-frame {\n  width: 100%;\n  max-width: 100%;\n}\n";

// src/components/scripts/utterances.inline.ts
var utterances_inline_default = 'var c=t=>(document.documentElement.getAttribute("saved-theme")??"light")==="dark"?t.dataset.darkTheme:t.dataset.lightTheme,s=t=>{let e=document.querySelector(".utterances-comments"),n=e?.querySelector(".utterances-frame");if(!e||!n?.contentWindow)return;let r=t?.detail?.theme??document.documentElement.getAttribute("saved-theme");n.contentWindow.postMessage({type:"set-theme",theme:r==="dark"?e.dataset.darkTheme:e.dataset.lightTheme},"https://utteranc.es")},a=()=>{let t=document.querySelector(".utterances-comments");if(!t)return;t.replaceChildren();let e=document.createElement("script");e.src="https://utteranc.es/client.js",e.async=!0,e.crossOrigin="anonymous",e.setAttribute("repo",t.dataset.repo),e.setAttribute("issue-term",t.dataset.issueTerm??"pathname"),e.setAttribute("theme",c(t)),t.dataset.label&&e.setAttribute("label",t.dataset.label),t.appendChild(e),document.addEventListener("themechange",s),window.addCleanup?.(()=>document.removeEventListener("themechange",s))};document.addEventListener("nav",a);document.addEventListener("render",a);\n';
var l;
function S(n2) {
  return n2.children;
}
l = { __e: function(n2, l2, u3, t2) {
  for (var i2, r2, o2; l2 = l2.__; ) if ((i2 = l2.__c) && !i2.__) try {
    if ((r2 = i2.constructor) && null != r2.getDerivedStateFromError && (i2.setState(r2.getDerivedStateFromError(n2)), o2 = i2.__d), null != i2.componentDidCatch && (i2.componentDidCatch(n2, t2 || {}), o2 = i2.__d), o2) return i2.__E = i2;
  } catch (l3) {
    n2 = l3;
  }
  throw n2;
} }, "function" == typeof Promise ? Promise.prototype.then.bind(Promise.resolve()) : setTimeout, Math.random().toString(8);

// node_modules/preact/jsx-runtime/dist/jsxRuntime.mjs
var f2 = 0;
function u2(e2, t2, n2, o2, i2, u3) {
  t2 || (t2 = {});
  var a2, c2, p2 = t2;
  if ("ref" in p2) for (c2 in p2 = {}, t2) "ref" == c2 ? a2 = t2[c2] : p2[c2] = t2[c2];
  var l2 = { type: e2, props: p2, key: n2, ref: a2, __k: null, __: null, __b: 0, __e: null, __c: null, constructor: void 0, __v: --f2, __i: -1, __u: 0, __source: i2, __self: u3 };
  if ("function" == typeof e2 && (a2 = e2.defaultProps)) for (c2 in a2) void 0 === p2[c2] && (p2[c2] = a2[c2]);
  return l.vnode && l.vnode(l2), l2;
}

// src/components/UtterancesComments.tsx
var UtterancesComments_default = ((opts) => {
  const options = {
    repo: "tom-jerr/blog-comments",
    issueTerm: "pathname",
    label: "comments",
    lightTheme: "github-light",
    darkTheme: "github-dark",
    ...opts
  };
  const UtterancesComments = ({
    displayClass,
    fileData
  }) => {
    const comments = fileData.frontmatter?.comments;
    if (comments === false || comments === "false") return /* @__PURE__ */ u2(S, {});
    return /* @__PURE__ */ u2(
      "div",
      {
        class: [displayClass, "utterances-comments"].filter(Boolean).join(" "),
        "data-repo": options.repo,
        "data-issue-term": options.issueTerm,
        "data-label": options.label,
        "data-light-theme": options.lightTheme,
        "data-dark-theme": options.darkTheme
      }
    );
  };
  UtterancesComments.css = utterances_default;
  UtterancesComments.afterDOMLoaded = utterances_inline_default;
  return UtterancesComments;
});

export { UtterancesComments_default as UtterancesComments };
//# sourceMappingURL=index.js.map
//# sourceMappingURL=index.js.map