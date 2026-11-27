# Product site only. Build with web/ as the context:
#   docker build -f web/Dockerfile -t ftaas-web web
# Or: railway up web --path-as-root --service ftaas-web
FROM node:22-alpine
WORKDIR /app
COPY package.json server.js layout.js pages.js catalog-data.js run-port.js ./
COPY public ./public
EXPOSE 3000
# Host-assigned PORT wins at runtime (see server.js).
CMD ["node", "server.js"]
