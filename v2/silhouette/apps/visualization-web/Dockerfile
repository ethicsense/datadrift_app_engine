FROM node:22-alpine AS build

WORKDIR /app

COPY apps/visualization-web/package.json /app/package.json
COPY apps/visualization-web/tsconfig.json /app/tsconfig.json
COPY apps/visualization-web/tsconfig.node.json /app/tsconfig.node.json
COPY apps/visualization-web/vite.config.ts /app/vite.config.ts
COPY apps/visualization-web/index.html /app/index.html
COPY apps/visualization-web/src /app/src

RUN npm install
RUN npm run build

FROM nginx:1.27-alpine

COPY apps/visualization-web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
