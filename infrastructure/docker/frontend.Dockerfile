# syntax=docker/dockerfile:1.7
# AgroSatCopilot — Frontend Nuxt 4 SSR multi-stage image
# Stages: deps (pnpm fetch) → dev (nuxt dev) → build (nuxt build) → runtime (.output)

# ----------------------------------------------------------------------------
# Stage 1: deps — descarga modulos en store pnpm (cacheable)
# ----------------------------------------------------------------------------
FROM node:20-alpine AS deps

ENV PNPM_HOME=/pnpm \
    PATH=/pnpm:$PATH \
    CI=true

RUN apk add --no-cache libc6-compat python3 make g++ \
    && corepack enable \
    && corepack prepare pnpm@9.12.3 --activate

WORKDIR /app
COPY package.json pnpm-lock.yaml* pnpm-workspace.yaml* ./
RUN --mount=type=cache,id=pnpm-store,target=/pnpm/store \
    pnpm fetch

# ----------------------------------------------------------------------------
# Stage 2: dev — modo desarrollo (volume bind-mount en docker-compose)
# ----------------------------------------------------------------------------
FROM node:20-alpine AS dev

ENV PNPM_HOME=/pnpm \
    PATH=/pnpm:$PATH \
    NODE_ENV=development \
    NUXT_HOST=0.0.0.0 \
    NUXT_PORT=3000

RUN apk add --no-cache libc6-compat python3 make g++ \
    && corepack enable \
    && corepack prepare pnpm@9.12.3 --activate

WORKDIR /app

COPY --from=deps /pnpm /pnpm
COPY package.json pnpm-lock.yaml* pnpm-workspace.yaml* ./
RUN --mount=type=cache,id=pnpm-store,target=/pnpm/store \
    pnpm install --frozen-lockfile --prefer-offline

EXPOSE 3000
CMD ["pnpm", "dev"]

# ----------------------------------------------------------------------------
# Stage 3: build — nuxt build genera .output/
# ----------------------------------------------------------------------------
FROM node:20-alpine AS build

ENV PNPM_HOME=/pnpm \
    PATH=/pnpm:$PATH \
    NODE_ENV=production \
    NITRO_PRESET=node-server

RUN apk add --no-cache libc6-compat python3 make g++ \
    && corepack enable \
    && corepack prepare pnpm@9.12.3 --activate

WORKDIR /app
COPY --from=deps /pnpm /pnpm
COPY package.json pnpm-lock.yaml* pnpm-workspace.yaml* ./
RUN --mount=type=cache,id=pnpm-store,target=/pnpm/store \
    pnpm install --frozen-lockfile --prefer-offline

COPY . .
RUN pnpm build

# ----------------------------------------------------------------------------
# Stage 4: runtime — slim image, sirve .output/server, USER non-root
# ----------------------------------------------------------------------------
FROM node:20-alpine AS runtime

ENV NODE_ENV=production \
    NUXT_HOST=0.0.0.0 \
    NUXT_PORT=3000 \
    PORT=3000

RUN apk add --no-cache libc6-compat tini \
    && addgroup --system --gid 1001 nuxt \
    && adduser --system --uid 1001 --ingroup nuxt nuxt

WORKDIR /app
COPY --from=build --chown=nuxt:nuxt /app/.output /app/.output

USER nuxt
EXPOSE 3000

ENTRYPOINT ["/sbin/tini", "--"]
CMD ["node", ".output/server/index.mjs"]
