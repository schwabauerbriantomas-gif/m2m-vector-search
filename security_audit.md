# 🔒 Auditoría de Seguridad - M2M Vector Search

**Generado:** 2026-03-17  
**Auditor:** MASFactory Security Team  
**Versión analizada:** 2.1.0  
**Alcance:** Todo el código fuente en `src/m2m/`, APIs, cluster

---

## Resumen de Severidad

| Severidad | Conteo |
|-----------|--------|
| 🔴 Critical | 2 |
| 🟠 High | 5 |
| 🟡 Medium | 7 |
| 🟢 Low | 4 |

---

## 🔴 CRITICAL

### C-01: APIs sin autenticación ni autorización
**Archivo:** `src/m2m/api/edge_api.py`, `src/m2m/api/coordinator_api.py`  
**Descripción:** Las APIs REST exponen endpoints de ingesta, búsqueda, eliminación y administración sin ningún mecanismo de autenticación. Cualquier nodo de red puede:
- Ingerir vectores arbitrarios
- Eliminar colecciones enteras
- Registrar nodos en el cluster
- Obtener métricas internas

**Impacto:** Denegación de servicio, corrupción de datos, exposición de información indexada.

**Recomendación:** Implementar API key, JWT o mTLS como mínimo. La clase `M2MClient` ya tiene soporte para `api_key` pero el servidor no lo valida.

### C-02: Comunicación entre nodos sin cifrado ni autenticación
**Archivo:** `src/m2m/cluster/`  
**Descripción:** Los nodos del cluster se comunican via HTTP plano sin TLS. Los mensajes de heartbeat, routing y sincronización viajan en texto claro, incluyendo vectores de embeddings (que pueden contener información sensible del dataset original).

**Impacto:** Man-in-the-middle attacks, interceptación de embeddings, spoofing de nodos.

**Recomendación:** Forzar HTTPS con certificados, implementar firma de mensajes entre nodos.

---

## 🟠 HIGH

### H-01: Path traversal en storage
**Archivo:** `src/m2m/storage.py`  
**Descripción:** `M2MPersistence` acepta `storage_path` de usuario sin sanitización. Si el path contiene `../`, puede escribir fuera del directorio previsto. Los nombres de colecciones pasados a endpoints API se usan directamente en rutas de archivo.

**Recomendación:** Validar y canonicalizar paths con `os.path.realpath()`, verificar que están dentro de un directorio raíz permitido.

### H-02: Input validation faltante en dimensiones
**Archivo:** `src/m2m/__init__.py` (SimpleVectorDB.add)  
**Descripción:** No se valida que `vectors.ndim == 2` o que `vectors.shape[1] == self.latent_dim` antes de procesar. Un vector con dimensiones incorrectas puede causar un crash silencioso o corrupción del índice.

**Impacto:** Crash del servicio, corrupción del índice HRM2.

### H-03: Denegación de servicio por ingesta masiva
**Archivo:** `src/m2m/api/edge_api.py`  
**Descripción:** No hay rate limiting en los endpoints de ingesta. Un atacante puede enviar millones de vectores para agotar memoria RAM/VRAM.

**Recomendación:** Implementar rate limiting por IP y por colección. Limitar el tamaño del body del request.

### H-04: Error messages exponen información interna
**Archivo:** `src/m2m/__init__.py`, múltiples módulos  
**Descripción:** Excepciones con stack traces completos se propagan a las respuestas HTTP. Ejemplo: `'M2MEngine' object has no attribute 'splats'` expone la estructura interna del código.

**Recomendación:** Implementar error handler global que devuelva mensajes genéricos y loggee los detalles internamente.

### H-05: Serialización pickle insegura
**Archivo:** `src/m2m/storage.py`  
**Descripción:** Si se usa pickle para serialización de datos, esto permite ejecución de código arbitrario si un atacante puede modificar los archivos de persistencia.

**Recomendación:** Usar formatos seguros (JSON, msgpack, numpy `.npy`). Si pickle es necesario, implementar firma HMAC.

---

## 🟡 MEDIUM

### M-01: CORS sin configuración explícita
**Archivo:** APIs FastAPI  
**Descripción:** Sin configuración CORS, las APIs aceptan requests de cualquier origen.

### M-02: UUIDs auto-generados predecibles (time-based)
**Archivo:** `src/m2m/__init__.py`  
**Descripción:** `uuid.uuid4()` es correcto (aleatorio), pero la función fallback o Legacy paths podrían usar IDs predecibles.

### M-03: Memoria ilimitada para embeddings
**Descripción:** No hay límite configurable en el número máximo de vectores por colección. Un usuario puede causar OOM.

### M-04: GPU memory exhaustion
**Archivo:** `src/m2m/splats.py`, `gpu_vector_index.py`  
**Descripción:** El GPUVectorIndex reserva memoria basada en `max_splats` sin verificar VRAM disponible real. Puede causar GPU OOM y crash del driver.

### M-05: SQL Injection (si se usa SQLite para metadata)
**Archivo:** `src/m2m/storage.py`  
**Descripción:** Si hay consultas SQL construidas por string concatenación (no verificado completamente), existe riesgo de inyección.

### M-06: Information disclosure en /health y /stats
**Archivo:** APIs  
**Descripción:** Los endpoints de health y stats exponen versión de Python, sistema operativo, hardware, tamaños de datasets, etc.

### M-07: Concurrent access sin locks
**Archivo:** `src/m2m/__init__.py`, `splats.py`  
**Descripción:** Operaciones de add/delete/search no están protegidas por locks. Acceso concurrente puede causar race conditions.

---

## 🟢 LOW

### L-01: Email del autor en código fuente
**Archivo:** `src/m2m/__init__.py`  
`schwabauerbriantomas@gmail.com` - podría ser harvesteado.

### L-02: Debug prints en producción
**Descripción:** Múltiples `print()` statements en el código. Deberían usar `logging` con niveles apropiados.

### L-03: Dependencias sin versiones fijas
**Archivo:** `pyproject.toml`  
**Descripción:** Algunas dependencias usan rangos amplios (ej. `numpy>=1.24`).

### L-04: Falta de type checking estricto
**Descripción:** No hay configuración de mypy o similar en el proyecto.

---

## Vulnerabilidades en Dependencias

No se detectaron CVEs críticos conocidos en las dependencias principales (numpy, fastapi, uvicorn, requests). Se recomienda ejecutar `pip audit` o `safety check` periódicamente.

---

## Plan de Remediación Priorizado

1. **Inmediato (C-01, C-02):** Agregar autenticación a APIs y cifrar comunicaciones entre nodos
2. **Corto plazo (H-01 a H-05):** Validación de inputs, rate limiting, error handling
3. **Mediano plazo (M-01 a M-07):** CORS, locks, límites de memoria, logging apropiado
4. **Largo plazo (L-01 a L-04):** Type checking, fijar dependencias, CI security scanning
