"""
图数据库客户端
支持Neo4j、ArangoDB等多种图数据库
用于存储和查询政治理论概念图谱
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class GraphDatabaseClient(ABC):
    """图数据库客户端抽象基类"""

    @abstractmethod
    def connect(self) -> bool:
        """连接到数据库"""
        pass

    @abstractmethod
    def disconnect(self):
        """断开数据库连接"""
        pass

    @abstractmethod
    def create_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """创建节点"""
        pass

    @abstractmethod
    def create_edge(self, from_id: str, to_id: str, edge_type: str, properties: Dict[str, Any]) -> bool:
        """创建边"""
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """获取节点"""
        pass

    @abstractmethod
    def get_neighbors(self, node_id: str, direction: str = "both") -> List[Dict[str, Any]]:
        """获取邻居节点"""
        pass

    @abstractmethod
    def search_nodes(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索节点"""
        pass

    @abstractmethod
    def get_subgraph(self, center_id: str, depth: int = 2) -> Dict[str, Any]:
        """获取子图"""
        pass

    @abstractmethod
    def batch_create_nodes(self, nodes: List[Tuple[str, Dict[str, Any]]]) -> bool:
        """批量创建节点"""
        pass

    @abstractmethod
    def batch_create_edges(self, edges: List[Tuple[str, str, str, Dict[str, Any]]]) -> bool:
        """批量创建边"""
        pass

    @abstractmethod
    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图统计信息"""
        pass


class Neo4jClient(GraphDatabaseClient):
    """Neo4j图数据库客户端"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.driver = None
        self.database = config.get('database', 'neo4j')

    def connect(self) -> bool:
        """连接到Neo4j数据库"""
        try:
            from neo4j import GraphDatabase

            self.driver = GraphDatabase.driver(
                self.config['uri'],
                auth=(self.config['username'], self.config['password'])
            )

            # 测试连接
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                result.single()

            logger.info(f"成功连接到Neo4j数据库: {self.config['uri']}")
            return True

        except ImportError:
            logger.error("Neo4j驱动未安装，请运行: pip install neo4j")
            return False
        except Exception as e:
            logger.error(f"连接Neo4j数据库失败: {e}")
            return False

    def disconnect(self):
        """断开Neo4j连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已断开")

    def create_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """创建Neo4j节点"""
        try:
            with self.driver.session(database=self.database) as session:
                query = """
                MERGE (n:Concept {id: $id})
                SET n += $properties
                SET n.updated_at = datetime()
                """
                session.run(query, id=node_id, properties=properties)
                return True
        except Exception as e:
            logger.error(f"创建Neo4j节点失败: {e}")
            return False

    def create_edge(self, from_id: str, to_id: str, edge_type: str, properties: Dict[str, Any]) -> bool:
        """创建Neo4j边"""
        try:
            with self.driver.session(database=self.database) as session:
                query = """
                MATCH (a:Concept {id: $from_id})
                MATCH (b:Concept {id: $to_id})
                MERGE (a)-[r:RELATION {type: $edge_type}]->(b)
                SET r += $properties
                SET r.updated_at = datetime()
                """
                session.run(query, from_id=from_id, to_id=to_id,
                           edge_type=edge_type, properties=properties)
                return True
        except Exception as e:
            logger.error(f"创建Neo4j边失败: {e}")
            return False

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """获取Neo4j节点"""
        try:
            with self.driver.session(database=self.database) as session:
                query = "MATCH (n:Concept {id: $id}) RETURN n"
                result = session.run(query, id=node_id)
                record = result.single()
                if record:
                    return dict(record['n'])
                return None
        except Exception as e:
            logger.error(f"获取Neo4j节点失败: {e}")
            return None

    def get_neighbors(self, node_id: str, direction: str = "both") -> List[Dict[str, Any]]:
        """获取Neo4j邻居节点"""
        try:
            with self.driver.session(database=self.database) as session:
                if direction == "outgoing":
                    query = """
                    MATCH (n:Concept {id: $id})-[r]->(m:Concept)
                    RETURN m, r.type as relation_type, properties(r) as properties
                    """
                elif direction == "incoming":
                    query = """
                    MATCH (n:Concept {id: $id})<-[r]-(m:Concept)
                    RETURN m, r.type as relation_type, properties(r) as properties
                    """
                else:  # both
                    query = """
                    MATCH (n:Concept {id: $id})-[r]-(m:Concept)
                    RETURN m, r.type as relation_type, properties(r) as properties
                    """

                result = session.run(query, id=node_id)
                neighbors = []
                for record in result:
                    neighbor_data = {
                        'node': dict(record['m']),
                        'relation_type': record['relation_type'],
                        'properties': dict(record['properties']) if record['properties'] else {}
                    }
                    neighbors.append(neighbor_data)
                return neighbors
        except Exception as e:
            logger.error(f"获取Neo4j邻居节点失败: {e}")
            return []

    def search_nodes(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索Neo4j节点"""
        try:
            with self.driver.session(database=self.database) as session:
                search_query = """
                MATCH (n:Concept)
                WHERE n.name CONTAINS $query OR n.definition CONTAINS $query
                RETURN n,
                       CASE
                           WHEN n.name STARTS WITH $query THEN 3
                           WHEN n.name CONTAINS $query THEN 2
                           WHEN n.definition CONTAINS $query THEN 1
                           ELSE 0
                       END as score
                ORDER BY score DESC
                LIMIT $limit
                """
                result = session.run(search_query, query=query, limit=limit)
                nodes = []
                for record in result:
                    node_data = dict(record['n'])
                    node_data['search_score'] = record['score']
                    nodes.append(node_data)
                return nodes
        except Exception as e:
            logger.error(f"搜索Neo4j节点失败: {e}")
            return []

    def get_subgraph(self, center_id: str, depth: int = 2) -> Dict[str, Any]:
        """获取Neo4j子图"""
        try:
            with self.driver.session(database=self.database) as session:
                query = """
                MATCH (center:Concept {id: $center_id})
                CALL apoc.path.subgraphAll(center, {
                    maxLevel: $depth,
                    relationshipFilter: "RELATION",
                    labelFilter: "+Concept"
                })
                YIELD nodes, relationships
                RETURN nodes, relationships
                """
                result = session.run(query, center_id=center_id, depth=depth)
                record = result.single()
                if record:
                    return {
                        'nodes': [dict(node) for node in record['nodes']],
                        'relationships': [dict(rel) for rel in record['relationships']]
                    }
                return {'nodes': [], 'relationships': []}
        except Exception as e:
            logger.error(f"获取Neo4j子图失败: {e}")
            # 如果APOC不可用，使用基础查询
            try:
                with self.driver.session(database=self.database) as session:
                    query = """
                    MATCH path = (center:Concept {id: $center_id})-[:RELATION*1..$depth]-(neighbor:Concept)
                    WITH collect(DISTINCT nodes(path)) as node_lists, collect(DISTINCT relationships(path)) as rel_lists
                    RETURN apoc.coll.flatten(node_lists) as nodes, apoc.coll.flatten(rel_lists) as relationships
                    """
                    result = session.run(query, center_id=center_id, depth=depth)
                    record = result.single()
                    if record:
                        return {
                            'nodes': [dict(node) for node in record['nodes']],
                            'relationships': [dict(rel) for rel in record['relationships']]
                        }
                    return {'nodes': [], 'relationships': []}
            except:
                logger.error("Neo4j子图查询完全失败，请检查APOC插件")
                return {'nodes': [], 'relationships': []}

    def batch_create_nodes(self, nodes: List[Tuple[str, Dict[str, Any]]]) -> bool:
        """批量创建Neo4j节点"""
        try:
            with self.driver.session(database=self.database) as session:
                query = """
                UNWIND $batch AS item
                MERGE (n:Concept {id: item.id})
                SET n += item.properties
                SET n.updated_at = datetime()
                """
                batch_data = [{'id': node_id, 'properties': properties} for node_id, properties in nodes]
                session.run(query, batch=batch_data)
                logger.info(f"批量创建 {len(nodes)} 个Neo4j节点成功")
                return True
        except Exception as e:
            logger.error(f"批量创建Neo4j节点失败: {e}")
            return False

    def batch_create_edges(self, edges: List[Tuple[str, str, str, Dict[str, Any]]]) -> bool:
        """批量创建Neo4j边"""
        try:
            with self.driver.session(database=self.database) as session:
                query = """
                UNWIND $batch AS item
                MATCH (a:Concept {id: item.from_id})
                MATCH (b:Concept {id: item.to_id})
                MERGE (a)-[r:RELATION {type: item.edge_type}]->(b)
                SET r += item.properties
                SET r.updated_at = datetime()
                """
                batch_data = [{'from_id': from_id, 'to_id': to_id, 'edge_type': edge_type, 'properties': properties}
                             for from_id, to_id, edge_type, properties in edges]
                session.run(query, batch=batch_data)
                logger.info(f"批量创建 {len(edges)} 个Neo4j边成功")
                return True
        except Exception as e:
            logger.error(f"批量创建Neo4j边失败: {e}")
            return False

    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取Neo4j图统计信息"""
        try:
            with self.driver.session(database=self.database) as session:
                stats = {}

                # 节点数量
                result = session.run("MATCH (n:Concept) RETURN count(n) as node_count")
                stats['node_count'] = result.single()['node_count']

                # 边数量
                result = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) as edge_count")
                stats['edge_count'] = result.single()['edge_count']

                # 平均度数
                if stats['node_count'] > 0:
                    stats['average_degree'] = (2 * stats['edge_count']) / stats['node_count']
                else:
                    stats['average_degree'] = 0

                # 连通分量数
                result = session.run("""
                CALL gds.wcc.stream('myGraph')
                YIELD componentId, componentSize
                RETURN count(DISTINCT componentId) as connected_components
                """)
                try:
                    stats['connected_components'] = result.single()['connected_components']
                except:
                    stats['connected_components'] = None  # GDS不可用

                return stats
        except Exception as e:
            logger.error(f"获取Neo4j图统计信息失败: {e}")
            return {}


class ArangoDBClient(GraphDatabaseClient):
    """ArangoDB图数据库客户端"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self.db = None

    def connect(self) -> bool:
        """连接到ArangoDB数据库"""
        try:
            from arango import ArangoClient

            self.client = ArangoClient(host=f"http://{self.config['host']}:{self.config['port']}")

            # 连接到系统数据库
            sys_db = self.client.db('_system', username=self.config['username'], password=self.config['password'])

            # 创建数据库（如果不存在）
            if not sys_db.has_database(self.config['database']):
                sys_db.create_database(self.config['database'])

            # 连接到目标数据库
            self.db = self.client.db(self.config['database'],
                                    username=self.config['username'],
                                    password=self.config['password'])

            # 创建集合（如果不存在）
            if not self.db.has_collection('concepts'):
                self.db.create_collection('concepts')

            if not self.db.has_collection('relations'):
                self.db.create_collection('relations', edge=True)

            logger.info(f"成功连接到ArangoDB数据库: {self.config['host']}:{self.config['port']}")
            return True

        except ImportError:
            logger.error("ArangoDB驱动未安装，请运行: pip install python-arango")
            return False
        except Exception as e:
            logger.error(f"连接ArangoDB数据库失败: {e}")
            return False

    def disconnect(self):
        """断开ArangoDB连接"""
        if self.client:
            self.client.close()
            logger.info("ArangoDB连接已断开")

    def create_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """创建ArangoDB节点"""
        try:
            concepts = self.db.collection('concepts')
            properties['_key'] = node_id
            concepts.insert(properties)
            return True
        except Exception as e:
            logger.error(f"创建ArangoDB节点失败: {e}")
            return False

    def create_edge(self, from_id: str, to_id: str, edge_type: str, properties: Dict[str, Any]) -> bool:
        """创建ArangoDB边"""
        try:
            relations = self.db.collection('relations')
            edge_data = {
                '_from': f'concepts/{from_id}',
                '_to': f'concepts/{to_id}',
                'type': edge_type,
                **properties
            }
            relations.insert(edge_data)
            return True
        except Exception as e:
            logger.error(f"创建ArangoDB边失败: {e}")
            return False

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """获取ArangoDB节点"""
        try:
            concepts = self.db.collection('concepts')
            if concepts.has(node_id):
                return concepts.get(node_id)
            return None
        except Exception as e:
            logger.error(f"获取ArangoDB节点失败: {e}")
            return None

    def get_neighbors(self, node_id: str, direction: str = "both") -> List[Dict[str, Any]]:
        """获取ArangoDB邻居节点"""
        try:
            bind_vars = {'start_node': f'concepts/{node_id}'}

            if direction == "outgoing":
                query = """
                FOR v, e IN 1..1 OUTBOUND @start_node relations
                RETURN {node: v, relation_type: e.type, properties: e}
                """
            elif direction == "incoming":
                query = """
                FOR v, e IN 1..1 INBOUND @start_node relations
                RETURN {node: v, relation_type: e.type, properties: e}
                """
            else:  # both
                query = """
                FOR v, e IN 1..1 ANY @start_node relations
                RETURN {node: v, relation_type: e.type, properties: e}
                """

            cursor = self.db.aql.execute(query, bind_vars=bind_vars)
            return list(cursor)
        except Exception as e:
            logger.error(f"获取ArangoDB邻居节点失败: {e}")
            return []

    def search_nodes(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索ArangoDB节点"""
        try:
            bind_vars = {'query': query, 'limit': limit}
            aql_query = """
            FOR doc IN concepts
            SEARCH doc.name LIKE @query OR doc.definition LIKE @query
            LET score =
                CASE
                    WHEN doc.name LIKE @query + '%' THEN 3
                    WHEN doc.name LIKE '%' + @query + '%' THEN 2
                    WHEN doc.definition LIKE '%' + @query + '%' THEN 1
                    ELSE 0
                END
            SORT score DESC
            LIMIT @limit
            RETURN MERGE(doc, {search_score: score})
            """
            cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
            return list(cursor)
        except Exception as e:
            logger.error(f"搜索ArangoDB节点失败: {e}")
            return []

    def get_subgraph(self, center_id: str, depth: int = 2) -> Dict[str, Any]:
        """获取ArangoDB子图"""
        try:
            bind_vars = {'start_node': f'concepts/{center_id}', 'depth': depth}
            query = """
            FOR v, e, p IN 1..@depth ANY @start_node relations
            COLLECT nodes = UNIQUE(p.vertices), edges = UNIQUE(p.edges)
            RETURN {
                nodes: nodes,
                relationships: edges
            }
            """
            cursor = self.db.aql.execute(query, bind_vars=bind_vars)
            result = list(cursor)
            return result[0] if result else {'nodes': [], 'relationships': []}
        except Exception as e:
            logger.error(f"获取ArangoDB子图失败: {e}")
            return {'nodes': [], 'relationships': []}

    def batch_create_nodes(self, nodes: List[Tuple[str, Dict[str, Any]]]) -> bool:
        """批量创建ArangoDB节点"""
        try:
            concepts = self.db.collection('concepts')
            documents = []
            for node_id, properties in nodes:
                doc = properties.copy()
                doc['_key'] = node_id
                documents.append(doc)

            concepts.import_many(documents)
            logger.info(f"批量创建 {len(nodes)} 个ArangoDB节点成功")
            return True
        except Exception as e:
            logger.error(f"批量创建ArangoDB节点失败: {e}")
            return False

    def batch_create_edges(self, edges: List[Tuple[str, str, str, Dict[str, Any]]]) -> bool:
        """批量创建ArangoDB边"""
        try:
            relations = self.db.collection('relations')
            documents = []
            for from_id, to_id, edge_type, properties in edges:
                doc = {
                    '_from': f'concepts/{from_id}',
                    '_to': f'concepts/{to_id}',
                    'type': edge_type,
                    **properties
                }
                documents.append(doc)

            relations.import_many(documents)
            logger.info(f"批量创建 {len(edges)} 个ArangoDB边成功")
            return True
        except Exception as e:
            logger.error(f"批量创建ArangoDB边失败: {e}")
            return False

    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取ArangoDB图统计信息"""
        try:
            stats = {}

            # 节点数量
            concepts = self.db.collection('concepts')
            stats['node_count'] = concepts.count()

            # 边数量
            relations = self.db.collection('relations')
            stats['edge_count'] = relations.count()

            # 平均度数
            if stats['node_count'] > 0:
                stats['average_degree'] = (2 * stats['edge_count']) / stats['node_count']
            else:
                stats['average_degree'] = 0

            return stats
        except Exception as e:
            logger.error(f"获取ArangoDB图统计信息失败: {e}")
            return {}


def get_graph_database_client(config_path: str = "config/config.yaml") -> Optional[GraphDatabaseClient]:
    """获取图数据库客户端实例"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        graph_config = config.get('graph_database', {})

        if not graph_config.get('enabled', False):
            logger.info("图数据库未启用，使用内存模式")
            return None

        db_type = graph_config.get('type', 'neo4j').lower()

        if db_type == 'neo4j':
            return Neo4jClient(graph_config['neo4j'])
        elif db_type == 'arangodb':
            return ArangoDBClient(graph_config['arangodb'])
        else:
            logger.error(f"不支持的图数据库类型: {db_type}")
            return None

    except Exception as e:
        logger.error(f"初始化图数据库客户端失败: {e}")
        return None


# 全局客户端实例
_graph_client_instance = None

def get_graph_client(config_path: str = "config/config.yaml") -> Optional[GraphDatabaseClient]:
    """获取全局图数据库客户端实例"""
    global _graph_client_instance
    if _graph_client_instance is None:
        _graph_client_instance = get_graph_database_client(config_path)
        if _graph_client_instance:
            _graph_client_instance.connect()
    return _graph_client_instance


def close_graph_client():
    """关闭全局图数据库客户端连接"""
    global _graph_client_instance
    if _graph_client_instance:
        _graph_client_instance.disconnect()
        _graph_client_instance = None