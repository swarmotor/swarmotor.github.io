# SWARMOTOR Homepage Design Contract

## Purpose

`index.html` is the primary personal homepage. It should communicate the resume essentials first, then route visitors to projects and writing. It is not a blog shelf.

## Visual Direction

- Preserve the existing SWARMOTOR resume/share style: warm off-white canvas, dark teal accent, light card surfaces, quiet borders, and an animated swarm/game background.
- Keep the page editorial and compact: high-signal resume blocks, no dense CV dump, no oversized marketing copy.
- Use the same floating controls as `share.html` and `posts.html`: language toggle, game toggle, and bottom-right page navigation.

## Tokens

- Background: `#fafaf9` with a low-opacity canvas layer.
- Surface: `#ffffff` and translucent white cards.
- Text: `#1a1a1a`, secondary `#525252`, muted `#737373`.
- Accent: `#0f4c5c`, accent light `#e8f4f6`.
- Warning/TODO: `#8a5a16` text on `#fff7e6`.
- Border: `#e5e5e5`.
- Shadow: low-opacity teal/black shadows only.
- Typography: `DM Sans` / `Noto Sans SC` for UI text, `Crimson Pro` for major headings.

## Homepage Sections

- Hero: name, role, location/contact affordances, and one concise positioning statement.
- Focus: three to four current research/engineering directions.
- Highlights: selected papers and projects from the resume.
- Skills: condensed technical stack grouped by purpose.
- Writing: secondary links to the technical blog articles.
- Contact: email, phone/WeChat from the resume source.

## Interaction

- Buttons and cards may lift on hover with `transform` and shadow only.
- Language switching uses the existing `[data-lang]` body class pattern.
- Game background remains decorative and must not block content or navigation.

## Responsive Rules

- Desktop: centered 900-1000px page with two-column highlight grids where useful.
- Mobile: single-column sections, fixed controls remain reachable, content width keeps 1rem side padding.

## Research Article Pages

- Article shell: centered `960px` reading column with a compact hero, semantic table of contents, main article, and references.
- Type scale: `12px` metadata, `14px` supporting text, `16px` body, `20px` subsection, `28px` section, and `44px` desktop title; titles reduce to `32px` on mobile.
- Spacing scale: use multiples of `4px`; article sections use `48px` vertical separation, cards use `24px` padding, and inline gaps use `8px` or `12px`.
- Article primitives: quiet bordered cards, teal-left callouts, equation blocks, figure placeholders, horizontally scrollable data tables, reference lists, and amber TODO markers.
- The table of contents may remain sticky on desktop, but must return to normal document flow on tablet and mobile.
- Equations must remain selectable text and must not require JavaScript or an external math renderer.
- Tables must preserve semantic headers and use overflow wrappers instead of shrinking text below the type scale.
- Print output removes navigation chrome and decorative backgrounds while preserving equations, tables, callouts, and references.
- Research pages use no animated decorative background; reading stability takes priority over the homepage swarm treatment.

## Accepted Debt

- This static site currently duplicates game/background code across pages. For this pass, preserve the duplication to avoid a larger refactor.
