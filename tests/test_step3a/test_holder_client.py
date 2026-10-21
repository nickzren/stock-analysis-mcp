"""The registered MCP tool carries the same holder evidence through every projection."""

import json
from datetime import datetime

import pytest
from fastmcp import Client

from stock_analysis import SERVER_VERSION, server
from tests.support import decision_matrix as dm
from tests.support import holder_matrix as hm


@pytest.mark.parametrize('name', ['sound_bullish', 'critical_runway', 'etf_spy'])
async def test_registered_analyze_carries_holder_without_snapshot_or_sizing_feedback(monkeypatch, name):
    case = hm.STOCK_CASES[name]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return case.now.astimezone(tz) if tz is not None else case.now.replace(tzinfo=None)

    monkeypatch.setattr(dm.orchestrator, 'datetime', Clock)
    reference = await hm.run_stock(name)
    expected = reference['decision_card']['long_term']['if_held']
    with dm.stubbed(case):
        async with Client(server.mcp) as client:
            for tier in ('decision', 'standard', 'full'):
                for arguments in ({'account_size': case.account_size}, {'profile': 'core'},
                                  {'profile': 'speculative', 'risk_per_trade_pct': '0.25'}):
                    result = await client.call_tool('analyze', {'symbol': 'ACME', 'detail': tier, **arguments})
                    assert not result.is_error
                    body = json.loads(result.content[0].text)
                    assert body['decision_card']['long_term']['if_held'] == expected
                    assert body['watchlist_snapshot'] == reference['watchlist_snapshot']
                    assert body['meta']['schema_version'] == '8'
                    assert body['meta']['server_version'] == SERVER_VERSION
