from typedconfig.config import Config, key, section, group_key
from typedconfig.source import IniFileConfigSource

@section('azure_ml')
class AzureMlConfig(Config):
    subscription_id=key(cast=str)
    resource_group_name=key(cast=str)
    workspace_name=key(cast=str)

class AppConfig(Config):
    azure_ml = group_key(AzureMlConfig)    
    
config = AppConfig()
config.add_source(IniFileConfigSource("config.cfg"))
config.read()