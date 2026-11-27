# Product site only. The Python gateway image stays Dockerfile at the repo root.
FROM node:22-alpine
WORKDIR /app
COPY package.json server.js layout.js pages.js catalog-data.js ./
COPY public ./public
ENV FTAAS_WEB_PORT=3000
EXPOSE 3000
CMD ["node", "server.js"]
