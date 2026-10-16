from comfy_api.latest import ComfyExtension, io

from .common import register_model_folder
from .detect import SceneCutDetect, SceneCutDetectAdvanced, SceneCutDetectorLoader, SceneCutRefine
from .files import SceneCutLoadScenes, SceneCutSaveScenes
from .routes import register_routes
from .validation import SceneCutValidate, SceneCutValidatorLoader
from .video import SceneCutLoadVideo
from .view import SceneCutContactSheet
from .viewer import SceneCutSelectedScenes, SceneCutViewer

register_model_folder()
register_routes()

NODES = [
    SceneCutLoadVideo,
    SceneCutDetect,
    SceneCutViewer,
    SceneCutSelectedScenes,
    SceneCutContactSheet,
    SceneCutSaveScenes,
    SceneCutLoadScenes,
    SceneCutDetectorLoader,
    SceneCutDetectAdvanced,
    SceneCutRefine,
    SceneCutValidatorLoader,
    SceneCutValidate,
]


class SceneCutExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return NODES


async def comfy_entrypoint() -> SceneCutExtension:
    return SceneCutExtension()
